#!/usr/bin/env python3
"""Train and evaluate model variants on a common flow split."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
INNER = ROOT
sys.path.insert(0, str(INNER))
from datasets.traffic_dataset import TrafficDataset, FeatureScaler, SEQ_CONT_IDX, collate_fn
from model.encoders.stat_groups import build_cic_groups, GROUP_NAMES
from model.traffic_cbm import build_model

INPUTS = ['x_stat', 'x_seq', 'attn_mask', 'x_bytes', 'bytes_mask']
DATASETS = ['CipherSpectrum-AES128', 'CSTNET-TLS13']
CONCEPTS = ['stat/'+n for n in GROUP_NAMES] + ['seq/'+n for n in ['volume','timing','direction','tcp']] + [f'bytes/pkt_{i}' for i in range(1,6)] + ['bytes/cross_front','bytes/cross_middle','bytes/cross_back']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def fit_array_scaler(values):
    x = np.asarray(values, dtype=np.float64)
    scaler = FeatureScaler(x.shape[-1])
    scaler.n = len(x)
    scaler.mean = x.mean(0)
    scaler.M2 = ((x-scaler.mean)**2).sum(0)
    scaler.finalize()
    return scaler


def prepare_cache(dataset, output, budget=8192):
    from sklearn.model_selection import StratifiedShuffleSplit
    source = INNER / 'data' / dataset
    ds = TrafficDataset(str(source/'test_ids.txt'), str(source), bytes_per_packet=True)
    available = set(ds.id2stat) & set(ds.id2seq) & set(ds.id2bytes)
    splits = {s:[x.strip() for x in (source/f'{s}_ids.txt').read_text().splitlines() if x.strip() in available]
              for s in ['train','val','test']}
    assert not (set(splits['train']) & (set(splits['val']) | set(splits['test'])))
    assert not (set(splits['val']) & set(splits['test']))
    original_train = len(splits['train'])
    if len(splits['train']) > budget:
        labels = [ds.label2id[x.split('/',1)[0]] for x in splits['train']]
        take,_ = next(StratifiedShuffleSplit(n_splits=1,train_size=budget,random_state=20261026).split(labels,labels))
        splits['train'] = [splits['train'][i] for i in sorted(take)]
    stats = np.stack([np.nan_to_num(ds.id2stat[f].copy(),nan=0,posinf=0,neginf=0) for f in splits['train']])
    seq = np.concatenate([ds.id2seq[f][:100,SEQ_CONT_IDX] for f in splits['train']])
    if not np.isfinite(seq).all():
        raise ValueError('Non-finite training sequence features')
    ds.stat_scaler, ds.seq_scaler = fit_array_scaler(stats), fit_array_scaler(seq)
    packed = {}
    for split, ids in splits.items():
        ds.flow_ids = ids
        pieces = {k:[] for k in INPUTS+['label']}
        for start in range(0,len(ids),256):
            batch = collate_fn([ds[i] for i in range(start,min(start+256,len(ids)))])
            for k in pieces:
                pieces[k].append(batch[k])
        packed[split] = {k:torch.cat(v) for k,v in pieces.items()}
        packed[split]['flow_ids'] = ids
        print(f'[cache] {dataset} {split}: {len(ids)}',flush=True)
    groups = build_cic_groups(ds.stat_feat_cols,list(range(len(ds.stat_feat_cols))))
    meta = dict(dataset=dataset, train_budget=budget, original_train=original_train,
                subset_seed=20261026, sizes={s:len(v) for s,v in splits.items()},
                stat_dim=ds.stat_dim, stat_columns=ds.stat_feat_cols, concept_groups=groups,
                class_mapping=ds.label2id, concepts=CONCEPTS,
                normalization='fit only on the selected training flows',
                split_hashes={s:hashlib.sha256('\n'.join(ids).encode()).hexdigest() for s,ids in splits.items()})
    output.mkdir(parents=True,exist_ok=True)
    dest = output / f'{dataset}.pt'
    if dest.exists():
        raise FileExistsError(dest)
    torch.save(dict(meta=meta,splits=packed,stat_scaler=ds.stat_scaler,seq_scaler=ds.seq_scaler),dest)
    (output/f'{dataset}.json').write_text(json.dumps(meta,indent=2))


class TrainableBatchedGA2M(nn.Module):
    """Same MLPs and independent dropout, with differentiable batched computation."""
    def __init__(self, head):
        super().__init__()
        self.dropout = head.main_effects[0][2].p
        for prefix,effects in [('main',head.main_effects),('pair',head.pair_effects)]:
            for name,index,attribute in [('w1',0,'weight'),('b1',0,'bias'),('w2',-1,'weight'),('b2',-1,'bias')]:
                self.register_parameter(prefix+'_'+name,nn.Parameter(torch.stack([getattr(e[index],attribute) for e in effects]).detach().clone()))
        self.register_buffer('pairs',torch.tensor(head.pair_indices,dtype=torch.long))
        self.bias = nn.Parameter(head.bias.detach().clone())

    def forward(self,x):
        main = torch.nn.functional.gelu(torch.einsum('bti,thi->bth',x.unsqueeze(-1),self.main_w1)+self.main_b1)
        pair = torch.nn.functional.gelu(torch.einsum('bti,thi->bth',x[:,self.pairs],self.pair_w1)+self.pair_b1)
        main = torch.nn.functional.dropout(main,self.dropout,training=self.training)
        pair = torch.nn.functional.dropout(pair,self.dropout,training=self.training)
        return self.bias + (torch.einsum('bth,toh->bto',main,self.main_w2)+self.main_b2).sum(1) + (torch.einsum('bth,toh->bto',pair,self.pair_w2)+self.pair_b2).sum(1)


class JointTransformer(nn.Module):
    def __init__(self,input_dim,length,layers,embedding=False):
        super().__init__()
        self.input = nn.Embedding(65540,64) if embedding else nn.Linear(input_dim,64)
        self.cls = nn.Parameter(torch.zeros(1,1,64))
        self.position = nn.Parameter(torch.zeros(1,length+1,64))
        block = nn.TransformerEncoderLayer(64,4,256,dropout=0.2,activation='gelu',batch_first=True)
        self.encoder = nn.TransformerEncoder(block,layers,enable_nested_tensor=False)

    def forward(self,x,mask):
        x = self.input(x)
        x = torch.cat([self.cls.expand(len(x),-1,-1),x],1)+self.position
        padding = torch.cat([torch.zeros((len(x),1),device=x.device,dtype=torch.bool),mask==0],1)
        return self.encoder(x,src_key_padding_mask=padding)[:,0]


class JointBlackBox(nn.Module):
    """Unstructured within each modality and a jointly mixed 18-D latent bottleneck."""
    def __init__(self,stat_dim,classes,target_parameters):
        super().__init__()
        layers=[]
        for a,b in zip([stat_dim,128,128,128],[128,128,128,64]):
            layers += [nn.Linear(a,b),nn.BatchNorm1d(b),nn.ReLU(),nn.Dropout(0.2)]
        self.stat = nn.Sequential(*layers)
        self.seq = JointTransformer(7,100,8)
        self.byte = JointTransformer(None,315,4,embedding=True)
        self.fusion = nn.Linear(192,18)
        body = sum(p.numel() for p in self.parameters())
        width = max(16,round((target_parameters-body-classes)/(19+classes)))
        self.head = nn.Sequential(nn.Linear(18,width),nn.GELU(),nn.Dropout(0.2),nn.Linear(width,classes))
        self.head_width=width

    def forward(self,x_stat,x_seq,attn_mask,x_bytes,bytes_mask):
        stat = self.stat(x_stat)
        seq = self.seq(x_seq,attn_mask)
        byte = self.byte(x_bytes.flatten(1),bytes_mask.flatten(1))
        latent = self.fusion(torch.cat([stat,seq,byte],1))
        return self.head(latent),{'c_final':latent}


def make_model(kind,meta):
    reference=build_model(num_classes=len(meta['class_mapping']),stat_dim=meta['stat_dim'],
             concept_groups=meta['concept_groups'],num_concepts=6,head='ga2m',fusion_mode='6+4+5+3',
             stat_hidden=(128,128,128,64),max_seq_len=100,seq_model_dim=64,seq_num_heads=4,seq_layers=2,
             seq_pooling='cls',bytes_model_dim=64,bytes_num_heads=4,bytes_layers=2,bytes_pooling='cls',
             bytes_per_packet=True,bytes_pkt_len=63,num_packets=5,num_cross=3,dropout=0.2)
    target=sum(p.numel() for p in reference.parameters())
    if kind=='structured':
        reference.head=TrainableBatchedGA2M(reference.head)
        model=reference
    else:
        model=JointBlackBox(meta['stat_dim'],len(meta['class_mapping']),target)
    count=sum(p.numel() for p in model.parameters())
    assert abs(count-target)/target < 0.01
    return model,dict(parameters=count,target_parameters=target,
                     relative_parameter_difference=(count-target)/target,
                     head_parameters=sum(p.numel() for p in model.head.parameters()),
                     body_parameters=count-sum(p.numel() for p in model.head.parameters()))


def input_batch(data,index):
    return {k:data[k][index] for k in INPUTS}


@torch.inference_mode()
def collect(model,data,batch_size=256):
    model.eval()
    logits,latent=[],[]
    for start in range(0,len(data['label']),batch_size):
        out,extras=model(**input_batch(data,slice(start,start+batch_size)))
        logits.append(out.detach())
        latent.append(extras['c_final'].detach())
    return torch.cat(logits),torch.cat(latent)


def gpu_metrics(logits,labels,classes):
    cm=torch.bincount(labels*classes+logits.argmax(1),minlength=classes**2).reshape(classes,classes).double()
    tp=cm.diagonal()
    p=tp/cm.sum(0).clamp(min=1)
    r=tp/cm.sum(1).clamp(min=1)
    f=2*p*r/(p+r).clamp(min=1e-12)
    return dict(n=len(labels),accuracy=float(tp.sum()/cm.sum()),macro_f1=float(f.mean()))


def train(cache_path,output,kind,seed,epochs=40,batch_size=256,patience=8):
    seed_all(seed)
    cache=torch.load(cache_path,map_location='cpu')
    meta=cache['meta']
    model,counts=make_model(kind,meta)
    model=model.cuda()
    data={s:{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in values.items()}
          for s,values in cache['splits'].items()}
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(kind=kind,seed=seed,max_epochs=epochs,batch_size=batch_size,patience=patience,
         lr=3e-4,weight_decay=1e-2,label_smoothing=0.05,pretraining=False,
         checkpoint_selection='validation macro-F1 only',cache=str(cache_path),data=meta,capacity=counts,
         torch=torch.__version__,gpu=torch.cuda.get_device_name(0),precision='float32')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-2)
    steps=math.ceil(len(data['train']['label'])/batch_size)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs*steps)
    criterion=nn.CrossEntropyLoss(label_smoothing=0.05)
    generator=torch.Generator().manual_seed(seed)
    best=-1.0
    stale=0
    history=[]
    for epoch in range(epochs):
        model.train()
        order=torch.randperm(len(data['train']['label']),generator=generator).cuda()
        loss_sum=0
        start=time.perf_counter()
        for offset in range(0,len(order),batch_size):
            take=order[offset:offset+batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits,_=model(**input_batch(data['train'],take))
            loss=criterion(logits,data['train']['label'][take])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum+=float(loss.detach())*len(take)
        val_logits,_=collect(model,data['val'])
        val=gpu_metrics(val_logits,data['val']['label'],len(meta['class_mapping']))
        row=dict(epoch=epoch+1,train_loss=loss_sum/len(order),val=val,seconds=time.perf_counter()-start)
        history.append(row)
        if val['macro_f1']>best+1e-6:
            best=val['macro_f1']
            stale=0
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch+1,
                            val=val,protocol=protocol),output/'best.pt')
        else:
            stale+=1
        (output/'history.json').write_text(json.dumps(history,indent=2))
        print(f"[train] {meta['dataset']} {kind} seed={seed} epoch={epoch+1} loss={row['train_loss']:.4f} val_f1={val['macro_f1']:.4f} best={best:.4f} seconds={row['seconds']:.1f}",flush=True)
        if stale>=patience:
            break
    checkpoint=torch.load(output/'best.pt',map_location='cpu')
    model.load_state_dict(checkpoint['model'],strict=True)
    saved={}
    for split in ['train','val','test']:
        logits,latent=collect(model,data[split])
        saved[split]=dict(logits=logits.cpu(),latent=latent.cpu(),labels=data[split]['label'].cpu(),
                          flow_ids=data[split]['flow_ids'])
    torch.save(saved,output/'representations.pt')
    result=dict(best_epoch=checkpoint['epoch'],epochs_run=len(history),best_validation=checkpoint['val'],
                test=gpu_metrics(saved['test']['logits'],saved['test']['labels'],len(meta['class_mapping'])),capacity=counts)
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print('[done] '+json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=['cache','train'],required=True)
    ap.add_argument('--dataset',choices=DATASETS)
    ap.add_argument('--cache',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model',choices=['structured','blackbox'])
    ap.add_argument('--seed',type=int,default=40)
    ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--budget',type=int,default=8192)
    ap.add_argument('--batch-size',type=int,default=256)
    args=ap.parse_args()
    if args.task=='cache':
        prepare_cache(args.dataset,args.output,args.budget)
    else:
        train(args.cache,args.output,args.model,args.seed,args.epochs,args.batch_size)

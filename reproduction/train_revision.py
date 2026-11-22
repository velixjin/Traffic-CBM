"""Isolated full-data revision runner, with explicit protocol and resumable state."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rebuttal_controls import INPUTS,seed_all,TrainableBatchedGA2M,gpu_metrics,make_model
from model.traffic_cbm import build_model
from checkpoint_io import atomic_torch_save


class HybridHead(nn.Module):
    def __init__(self,structured,dimensions,classes):
        super().__init__()
        self.structured=structured
        self.residual=nn.Sequential(nn.Linear(dimensions,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,classes))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self,x):
        return self.structured(x)+self.residual(x)


def construct(kind,meta,pretrained=True,architecture=None,pretrain_path=None,dropout=.2):
    architecture=architecture or meta.get('byte_architecture',{})
    classes=len(meta['class_mapping'])
    loading={}
    if kind=='blackbox':
        if pretrained:raise ValueError('Use --scratch for the parameter-matched black box.')
        return make_model('blackbox',meta)[0],{}
    if kind=='blackbox_ga2m':
        # 2x2 control: unconstrained 18-D latent + the same GA2M fusion head
        if pretrained:raise ValueError('Use --scratch for the parameter-matched black box.')
        from model.traffic_cbm import GA2MHead
        model,_=make_model('blackbox',meta)
        model.head=TrainableBatchedGA2M(GA2MHead(18,classes))
        total=sum(p.numel() for p in model.parameters())
        return model,dict(note='unconstrained 18-D latent + GA2M head',parameters=total)
    model=build_model(num_classes=classes,stat_dim=meta['stat_dim'],concept_groups=meta['concept_groups'],
        num_concepts=6,head='mlp' if kind=='mlp' else 'ga2m',fusion_mode='6+4+5+3',
        stat_hidden=(128,128,128,64),max_seq_len=100,seq_model_dim=64,seq_num_heads=4,seq_layers=2,
        seq_pooling='cls',bytes_model_dim=64,bytes_num_heads=4,bytes_layers=2,bytes_pooling='cls',
        bytes_per_packet=True,bytes_pkt_len=architecture.get('tokens',63),
        num_packets=architecture.get('packets',5),num_cross=architecture.get('cross',3),dropout=dropout)
    if pretrained:
        if pretrain_path is None:
            raise ValueError('Supply --pretrain-ckpt or use --scratch.')
        path=Path(pretrain_path)
        state=torch.load(path,map_location='cpu')
        loading=dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),modules={})
        for name in ['stat_encoder','seq_encoder','bytes_encoder']:
            if name not in state:continue
            module=getattr(model,name);target=module.state_dict()
            compatible={k:v for k,v in state[name].items() if k in target and v.shape==target[k].shape}
            msg=module.load_state_dict(compatible,strict=False)
            loading['modules'][name]=dict(missing=msg.missing_keys,unexpected=msg.unexpected_keys,
                skipped_incompatible=[k for k in state[name] if k not in compatible])
    if kind=='mlp_matched':
        target=sum(p.numel() for p in model.head.parameters())
        width=max(1,round((target-classes)/(model.fused_dim+1+classes)))
        model.head=nn.Sequential(nn.Linear(model.fused_dim,width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,classes))
        actual=sum(p.numel() for p in model.head.parameters())
        loading['head_capacity_match']=dict(reference_ga2m_head_parameters=target,mlp_head_parameters=actual,
            mlp_width=width,relative_difference=(actual-target)/target,selection='Analytic parameter matching only; no held-out tuning')
    elif kind!='mlp':
        model.head=TrainableBatchedGA2M(model.head)
    if kind=='hybrid':
        model.head=HybridHead(model.head,model.fused_dim,classes)
    return model,loading


def forward(model,data,index):
    return model(**{k:data[k][index] for k in INPUTS})


@torch.inference_mode()
def collect(model,data,batch=256):
    model.eval();logits=[];latent=[]
    for start in range(0,len(data['label']),batch):
        scores,extra=forward(model,data,slice(start,start+batch))
        logits.append(scores.detach())
        if extra is not None:
            latent.append(extra['c_final'].detach())
    return torch.cat(logits),torch.cat(latent) if latent else None


def optimizer_for(model,lr=None,wd=None):
    # Command-line values override the default optimiser settings.
    rate=lr if lr else .0003
    decay=wd if wd else .01
    return torch.optim.AdamW(model.parameters(),lr=rate,weight_decay=decay),rate


def save_resume(path,model,optimizer,epoch,best,stale,history,protocol):
    atomic_torch_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,
        stale=stale,history=history,protocol=protocol,random=random.getstate(),numpy=np.random.get_state(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)



def augment_batch(data,take,args,generator):
    """Training-time input perturbations; validation/test stay clean."""
    batch={k:data[k][take] for k in INPUTS}
    if args.augment_stat:
        noise=torch.randn(batch['x_stat'].shape,device=batch['x_stat'].device)
        batch['x_stat']=batch['x_stat']+args.augment_stat*noise
    if args.augment_bytes:
        drop=torch.rand(batch['x_bytes'].shape,device=batch['x_bytes'].device)<args.augment_bytes
        batch['x_bytes']=torch.where(drop,torch.full_like(batch['x_bytes'],65536),batch['x_bytes'])
        batch['bytes_mask']=torch.where(drop,torch.zeros_like(batch['bytes_mask']),batch['bytes_mask'])
    if args.augment_seq:
        drop=torch.rand(batch['x_seq'].shape[:2],device=batch['x_seq'].device)<args.augment_seq
        keep=(~drop).unsqueeze(-1)
        batch['x_seq']=batch['x_seq']*keep
        batch['attn_mask']=batch['attn_mask']*(~drop).float()
    rates=getattr(args,'branch_rates',None)
    anneal=getattr(args,'augment_anneal',None)
    active=bool(rates) or (getattr(args,'augment_branch_drop',0.) and (getattr(args,'augment_drop_until',None) is None
            or getattr(args,'current_epoch',0)<args.augment_drop_until))
    if active:
        n=batch['x_stat'].shape[0];device=batch['x_stat'].device
        schedule=1.0
        if anneal:
            schedule=max(0.0,(args.epochs-getattr(args,'current_epoch',0))/max(anneal,1))
            schedule=min(1.0,schedule)
        keys=(('x_stat',None),('x_seq','attn_mask'),('x_bytes','bytes_mask'))
        for i,(key,mask_key) in enumerate(keys):
            base=rates[i] if rates else args.augment_branch_drop
            prob=base*(schedule if anneal else 1.0)
            if prob<=0: continue
            hit=torch.rand(n,device=device)<prob
            if hit.any():
                batch[key]=batch[key].clone();batch[key][hit]=0
                if mask_key:
                    batch[mask_key]=batch[mask_key].clone();batch[mask_key][hit]=0
    return batch


def train(args):
    if getattr(args,'pairwise_cross',False):
        import os as _os; _os.environ['CBM_PAIRWISE_CROSS']='1'
    if getattr(args,'branch_rates',None) and isinstance(args.branch_rates,str):
        args.branch_rates=[float(v) for v in args.branch_rates.split(',')]
    seed_all(args.seed)
    cache=torch.load(args.cache,map_location='cpu');meta=cache['meta']
    model,loading=construct(args.model,meta,not args.scratch,pretrain_path=args.pretrain_ckpt,dropout=args.dropout)
    model=model.cuda()
    teacher=None
    if getattr(args,'distill_from',None):
        from pathlib import Path as _P
        _dir=_P(args.distill_from)
        _proto=json.loads((_dir/'protocol.json').read_text())
        _kind=_proto.get('kind',_proto.get('model'))
        teacher,_=construct(_kind,meta,pretrained=False)
        teacher.load_state_dict(torch.load(_dir/'best.pt',map_location='cpu',weights_only=False)['model'])
        teacher=teacher.cuda().eval()
        for _p in teacher.parameters():_p.requires_grad_(False)
    fields=INPUTS+['label']
    data={s:{k:v.cuda() if k in fields and isinstance(v,torch.Tensor) else v for k,v in values.items()}
          for s,values in cache['splits'].items()}
    if getattr(args,'mask_branch',None):
        # Information ablation: the branch stays in the architecture but carries no
        # information in any split, so the run measures the remaining sources only.
        key={'stat':'x_stat','seq':'x_seq','bytes':'x_bytes'}[args.mask_branch]
        for values in data.values():
            if key in values:
                values[key]=torch.zeros_like(values[key])
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'result.json').exists():
        raise FileExistsError('Completed result exists; do not overwrite: '+str(args.output))
    optimizer,base_lr=optimizer_for(model,args.lr,args.weight_decay)
    protocol=dict(model=args.model,seed=args.seed,cache=str(args.cache),data=meta,epochs=args.epochs,
        patience=args.patience,batch_size=args.batch_size,pretrained=not args.scratch,pretrained_loading=loading,
        selection='validation macro-F1 only; test after checkpoint selection',precision='float32',
        optimizer='AdamW',base_lr=base_lr,label_smoothing=.05,
        schedule='stepwise cosine',
        parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        mask_branch=getattr(args,'mask_branch',None),
        adversarial_training=getattr(args,'adv_train_eps',0.) or None,
        distillation=dict(teacher=str(getattr(args,'distill_from',None)),
            alpha=getattr(args,'distill_alpha',0.5),temperature=getattr(args,'distill_temp',2.0))
            if getattr(args,'distill_from',None) else None,
        augment=dict(stat=getattr(args,'augment_stat',0.),bytes=getattr(args,'augment_bytes',0.),
            seq=getattr(args,'augment_seq',0.),branch_drop=getattr(args,'augment_branch_drop',0.)))
    start_epoch=0;best=-1.;stale=0;history=[]
    if (args.output/'resume.pt').exists():
        state=torch.load(args.output/'resume.pt',map_location='cuda')
        recorded_protocol = dict(state['protocol'])
        recorded_protocol.pop('limitations', None)
        assert recorded_protocol == protocol
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch'];best=state['best'];stale=state['stale'];history=state['history']
        random.setstate(state['random']);np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    elif (args.output/'protocol.json').exists():
        raise RuntimeError('Existing incomplete run lacks resumable state; use a new attempt directory')
    else:
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
        save_resume(args.output/'resume.pt',model,optimizer,0,best,stale,history,protocol)
    steps=math.ceil(len(data['train']['label'])/args.batch_size)
    criterion=nn.CrossEntropyLoss(label_smoothing=protocol['label_smoothing'])
    for epoch in range(start_epoch,args.epochs):
        args.current_epoch=epoch
        model.train();loss_sum=0;start=time.perf_counter()
        generator=torch.Generator().manual_seed(args.seed*1000+epoch)
        order=torch.randperm(len(data['train']['label']),generator=generator).cuda()
        for step,offset in enumerate(range(0,len(order),args.batch_size)):
            progress=epoch+step/steps
            lr=1e-6+(base_lr-1e-6)*.5*(1+math.cos(math.pi*progress/args.epochs))
            for group in optimizer.param_groups:
                group['lr']=lr*group.get('lr_scale',1.)
            take=order[offset:offset+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            _eps=getattr(args,'adv_train_eps',0.)
            if _eps:
                _clean={k:data['train'][k][take].clone() for k in INPUTS}
                _labels=data['train']['label'][take]
                _adv={k:v.clone() for k,v in _clean.items()}
                _adv['x_stat'].requires_grad_(True)
                _s,_=forward(model,_adv,slice(None))
                _grad,=torch.autograd.grad(criterion(_s,_labels),_adv['x_stat'])
                _adv={k:v.detach() for k,v in _adv.items()}
                _adv['x_stat']=(_adv['x_stat']+_eps*_grad.sign()).detach()
                _adv_scores,_=forward(model,_adv,slice(None))
                _clean_scores,_=forward(model,_clean,slice(None))
                scores=(_clean_scores+_adv_scores)/2  # placeholder, replaced below
                del scores
                loss=0.5*criterion(_clean_scores,_labels)+0.5*criterion(_adv_scores,_labels)
            elif args.augment_stat or args.augment_bytes or args.augment_seq or getattr(args,'augment_branch_drop',0.):
                batch=augment_batch(data['train'],take,args,generator)
                scores,_=forward(model,batch,slice(None))
            else:
                scores,_=forward(model,data['train'],take)
            if not getattr(args,'adv_train_eps',0.):
                loss=criterion(scores,data['train']['label'][take])
            if teacher is not None:
                with torch.no_grad():
                    teacher_scores,_=forward(teacher,data['train'],take)
                _T=getattr(args,'distill_temp',2.0)
                _kl=torch.nn.functional.kl_div(torch.log_softmax(scores/_T,dim=1),
                    torch.softmax(teacher_scores/_T,dim=1),reduction='batchmean')*_T*_T
                loss=loss+getattr(args,'distill_alpha',0.5)*_kl
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(take)
        val_logits,_=collect(model,data['val'],args.batch_size)
        val=gpu_metrics(val_logits,data['val']['label'],len(meta['class_mapping']))
        row=dict(epoch=epoch+1,train_loss=loss_sum/len(order),val=val,seconds=time.perf_counter()-start)
        history.append(row)
        if val['macro_f1']>best+1e-6:
            best=val['macro_f1'];stale=0
            atomic_torch_save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},protocol=protocol,
                epoch=epoch+1,val=val),args.output/'best.pt')
        else:
            stale+=1
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
        save_resume(args.output/'resume.pt',model,optimizer,epoch+1,best,stale,history,protocol)
        print(f"[epoch] {meta['dataset']} {args.model} seed={args.seed} {epoch+1}/{args.epochs} val_macro_f1={val['macro_f1']:.5f} time={row['seconds']:.1f}s",flush=True)
        if stale>=args.patience:
            break
    checkpoint=torch.load(args.output/'best.pt',map_location='cpu');model.load_state_dict(checkpoint['model'],strict=True)
    saved={}
    for split in ['train','val','test']:
        scores,latent=collect(model,data[split],args.batch_size)
        saved[split]=dict(logits=scores.cpu(),labels=data[split]['label'].cpu(),
            flow_ids=data[split].get('flow_ids',[]))
        if latent is not None:
            saved[split]['latent']=latent.cpu()
    torch.save(saved,args.output/'representations.pt')
    result=dict(best_epoch=checkpoint['epoch'],epochs_run=len(history),best_validation=checkpoint['val'],
        test=gpu_metrics(saved['test']['logits'],saved['test']['labels'],len(meta['class_mapping'])),protocol=protocol)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print('[done]',json.dumps(result['test']),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True,type=Path)
    ap.add_argument('--model',required=True,choices=['ga2m','mlp','mlp_matched','hybrid','blackbox','blackbox_ga2m'])
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--patience',type=int,default=10)
    ap.add_argument('--batch-size',type=int,default=256);ap.add_argument('--scratch',action='store_true')
    ap.add_argument('--augment-branch-rates',dest='branch_rates',type=str,default=None,
        help='per-branch dropout probabilities stat,seq,bytes, e.g. 0.05,0.05,0.20')
    ap.add_argument('--augment-anneal',dest='augment_anneal',type=int,default=None,
        help='linearly decay the dropout probability over the final N epochs')
    ap.add_argument('--augment-drop-until',dest='augment_drop_until',type=int,default=None,
        help='stop modality dropout after this many epochs, then finish clean')
    ap.add_argument('--augment-branch-drop',dest='augment_branch_drop',type=float,default=0.,
        help='per-sample probability of zeroing one entire evidence branch (modality dropout)')
    ap.add_argument('--augment-stat',dest='augment_stat',type=float,default=0.,
        help='training-time Gaussian noise sigma on standardised statistics')
    ap.add_argument('--augment-bytes',dest='augment_bytes',type=float,default=0.,
        help='training-time byte-token masking probability')
    ap.add_argument('--augment-seq',dest='augment_seq',type=float,default=0.,
        help='training-time packet dropout probability')
    ap.add_argument('--mask-branch',dest='mask_branch',choices=['stat','seq','bytes'],default=None,
        help='information ablation: zero one evidence source in every split')
    ap.add_argument('--lr',type=float,default=None,
        help='Initial learning rate (default: 0.0003)')
    ap.add_argument('--weight-decay',type=float,dest='weight_decay',default=None,
        help='weight-decay override; default keeps the paper value 1e-2')
    ap.add_argument('--dropout',type=float,default=0.2,
        help='dropout override; default keeps the paper value 0.2')
    ap.add_argument('--adv-train-eps',dest='adv_train_eps',type=float,default=0.,
        help='FGSM adversarial-training budget on the statistics branch (0 disables)')
    ap.add_argument('--distill-from',dest='distill_from',type=Path,default=None,
        help='teacher run directory; adds a KL term to the student objective')
    ap.add_argument('--distill-alpha',dest='distill_alpha',type=float,default=0.5)
    ap.add_argument('--distill-temp',dest='distill_temp',type=float,default=2.0)
    ap.add_argument('--pairwise-cross',dest='pairwise_cross',action='store_true',
        help='use the explicit pairwise-interaction cross-packet encoder')
    ap.add_argument('--pretrain-ckpt',type=Path)
    train(ap.parse_args())

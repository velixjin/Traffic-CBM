"""Equivalent masked-only reconstruction projection; original module stays read-only."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train.pretrain_seq_bytes import SeqBytesPretrainer,TOKEN_MASK
from model.encoders.sequence_encoder import SeqConceptEncoder


class MaskedProjectionPretrainer(SeqBytesPretrainer):
    def forward(self,batch,seq_mask_ratio=.15,bytes_mask_ratio=.15):
        x_seq=batch['x_seq'].float();attn=batch['attn_mask'].float()
        x_bytes=batch['x_bytes'].long();mask_bytes=batch['bytes_mask'].long()
        sequence_loss=torch.zeros((),device=x_seq.device)
        byte_loss=torch.zeros((),device=x_seq.device)
        for head,((indices,_),encoder) in zip(self.seq_recon_heads,zip(SeqConceptEncoder.GROUPS,self.seq_encoder.group_encoders)):
            group=x_seq[:,:,indices];chosen=(torch.rand_like(attn)<seq_mask_ratio)&(attn>0)
            altered=group.clone();altered[chosen]=0
            hidden=self._encode_group_tokens(encoder,altered,attn)
            predicted=head(hidden)
            if chosen.any():sequence_loss=sequence_loss+F.mse_loss(predicted[chosen],group[chosen])
        chosen=(torch.rand_like(mask_bytes.float())<bytes_mask_ratio)&(mask_bytes>0)
        altered=x_bytes.clone();altered[chosen]=TOKEN_MASK
        b,n,length=x_bytes.shape
        hidden=self._encode_bytes_packet_tokens(altered.reshape(b*n,length),mask_bytes.reshape(b*n,length))
        flat=chosen.reshape(b*n,length)
        if flat.any():
            byte_loss=byte_loss+F.cross_entropy(self.bytes_recon_head(hidden[flat]),x_bytes.reshape(b*n,length)[flat])
        groups=self.bytes_encoder.num_cross;width=self.bytes_encoder.group_len
        view=altered.reshape(b,n,groups,width);m=mask_bytes.reshape(b,n,groups,width)
        targets=x_bytes.reshape(b,n,groups,width);selection=chosen.reshape(b,n,groups,width)
        losses=[]
        for g in range(groups):
            z=view[:,:,g,:].reshape(b,n*width);valid=m[:,:,g,:].reshape(b,n*width)
            pick=selection[:,:,g,:].reshape(b,n*width)
            hidden=self._encode_bytes_cross_tokens(z,valid)
            if pick.any():
                losses.append(F.cross_entropy(self.bytes_recon_head(hidden[pick]),targets[:,:,g,:].reshape(b,n*width)[pick]))
        if losses:byte_loss=byte_loss+torch.stack(losses).mean()
        return sequence_loss+byte_loss,dict(seq_loss=float(sequence_loss.detach()),bytes_loss=float(byte_loss.detach()))

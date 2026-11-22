"""Publish new experiment checkpoints atomically, retaining the previous valid one."""
from pathlib import Path
import torch


def atomic_torch_save(value,path):
    path=Path(path)
    pending=path.with_name(path.name+'.pending')
    torch.save(value,pending)
    pending.replace(path)

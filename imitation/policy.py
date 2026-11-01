"""Four *independent* learned actors, no centralized allocation.

Each actor sees only features constructed by imitation.features. The physical
plant applies the four independent commands and retains safety protections.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .features import N_FEATURES, FEATURE_NAMES


class TugActor(nn.Module):
    def __init__(self,width=128):
        super().__init__()
        self.backbone=nn.Sequential(
            nn.Linear(N_FEATURES,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),
            nn.Linear(width,2))
    def forward(self,x):
        raw=self.backbone(x)
        return torch.stack([torch.sigmoid(raw[...,0]),torch.tanh(raw[...,1])],dim=-1)


class IndependentActors:
    """Loads all four students; one observation in, one action out, per agent."""
    def __init__(self,checkpoint_dir,device='cpu'):
        self.root=Path(checkpoint_dir);self.device=torch.device(device)
        info=json.loads((self.root/'actor_metadata.json').read_text())
        if info['feature_names']!=list(FEATURE_NAMES):
            raise ValueError('Feature schema mismatch. Regenerate the training data/checkpoints.')
        self.mean=np.asarray(info['mean'],dtype=np.float32)
        self.std=np.asarray(info['std'],dtype=np.float32)
        self.actors=[];self.width=int(info['width'])
        for i in range(1,5):
            actor=TugActor(width=self.width).to(self.device)
            state=torch.load(self.root/f'actor_{i}.pt',map_location=self.device,weights_only=True)
            actor.load_state_dict(state);actor.eval();self.actors.append(actor)

    def act(self,i,feature,previous_action=(0.,0.),slew_limiter=True):
        i=int(i)
        if not 1<=i<=4:raise ValueError('agent must be 1..4')
        x=np.asarray(feature,dtype=np.float32)
        if x.shape!=(N_FEATURES,) or not np.isfinite(x).all():raise ValueError('Bad actor observation')
        z=np.clip((x-self.mean)/self.std,-10.,10.)
        with torch.no_grad():
            pred=self.actors[i-1](torch.from_numpy(z).to(self.device)).cpu().numpy()
        fraction=float(np.clip(pred[0],0.,.90 if i<3 else .65))
        offset=float(np.clip(pred[1]*20.,-20.,20.))
        if slew_limiter:
            prev=np.asarray(previous_action,float)
            fraction=float(np.clip(fraction,max(0.,prev[0]-.38),prev[0]+.38))
            # replicate the teacher's 24-deg first-step offset slew bound
            if prev[0]>=.06:offset=float(np.clip(offset,prev[1]-24.,prev[1]+24.))
        return np.array([fraction,offset],dtype=np.float32)

    @staticmethod
    def to_env_action(command):
        fraction,offset=np.asarray(command,float)
        return np.array([2.*fraction-1.,offset/20.],dtype=np.float32)

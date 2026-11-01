"""Nonparametric per-tug supervised imitation baseline (no NMPC calls at rollout).

Designed as a diagnostic: every tug learns only its own (feature,action) pairs
from training scenarios. Evaluation never reads validation or test labels.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .features import N_FEATURES

class RetrievalActors:
    def __init__(self,dataset,neighbors=3,blend=1.0):
        data=np.load(dataset)
        tr=data['split']=='train'
        x=data['X'][tr];y=data['Y'][tr];ids=data['agent'][tr]
        self.blend=blend;self.neighbors=neighbors
        self.trees=[];self.y=[];self.mu=[];self.scale=[];self.baseline=[]
        # Fixed schema feature weights: prevent irrelevant cosine near-zero winds from dominating.
        w=np.array([2.7,3.2,1.8,.3,2.1,2.8,1.8,.65,.65,.5,.5,.10,.10,.10,
                    .3,1.5,1.,1.,1.8,1.,1.,.7,.15],np.float32)
        for i in range(1,5):
            z=x[ids==i].astype(np.float64)
            mu=z.mean(0)
            scale=np.maximum(z.std(0),.06)
            # scale wind/current dimensions as physical normalized features, not global s.d. of only training env
            scale[15:21]=np.maximum(scale[15:21],.40)
            if i>2:
                # For tow, normalized previous action often zero, don't let azimuth history dominate.
                w0=w.copy();w0[-1]=.07
            else:w0=w
            k=(z-mu)/scale*w0
            self.trees.append((cKDTree(k),w0))
            self.y.append(y[ids==i].astype(np.float64));self.mu.append(mu);self.scale.append(scale)
    def act(self,i,feature,previous_action=(0.,0.),slew_limiter=True):
        j=i-1
        tree,w=self.trees[j]
        query=(np.asarray(feature,dtype=np.float64)-self.mu[j])/self.scale[j]*w
        dist,idx=tree.query(query,k=self.neighbors)
        dist=np.atleast_1d(dist);idx=np.atleast_1d(idx)
        weights=1/np.maximum(.004,dist)**2
        y=(self.y[j][idx]*weights[:,None]).sum(0)/weights.sum()
        frac=float(np.clip(y[0],0.,.9 if i<3 else .65));off=float(np.clip(y[1]*20.,-20.,20.))
        if frac<.018:frac=0.
        if slew_limiter:
            prev=np.asarray(previous_action,float)
            frac=float(np.clip(frac,max(0.,prev[0]-.38),prev[0]+.38))
            if prev[0]>=.06:off=float(np.clip(off,prev[1]-24.,prev[1]+24.))
        return np.array([frac,off],np.float32)
    @staticmethod
    def to_env_action(command):
        frac,off=map(float,command)
        return np.array([2*frac-1,off/20.],np.float32)

"""Environment-conditioned routing between *independent* local neural actors.

For each tug: both candidate networks read exactly the same local observation;
a deterministic gate decides which network to use from wind/current projected
onto measured berth-error direction. No peer states, commands, global allocator
or runtime teacher query. The gate is an empirical model-selection heuristic;
not a proof of robustness, not itself a neural network, and not pure single-actor BC.
"""
from __future__ import annotations
import math
import numpy as np
from .policy import IndependentActors


def toward_load_score(feature,wind_weight=1.,current_weight=20.):
    x=np.asarray(feature,float)
    if x.shape!=(23,):raise ValueError('Expected local 23-component observation')
    err=x[:2]
    norm=float(np.linalg.norm(err))
    if norm<1e-7:return 0.
    # Goal heading is zero in the evaluated scenarios. Relative ship heading
    # rotates world disturbance vectors to the current ship frame.
    psi_rel=math.atan2(x[2],x[3]);c,s=math.cos(psi_rel),math.sin(psi_rel)
    wind=np.array([10*x[15]*x[16],10*x[15]*x[17]])
    cur=np.array([.5*x[18]*x[19],.5*x[18]*x[20]])
    def to_body(v):return np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1]])
    return float((wind_weight*to_body(wind)+current_weight*to_body(cur))@(err/norm))


class ContextRoutedActors:
    def __init__(self,standard_dir,adverse_dir,device='cpu',threshold=2.):
        self.standard=IndependentActors(standard_dir,device=device)
        self.adverse=IndependentActors(adverse_dir,device=device)
        self.threshold=float(threshold)
        self.counts=dict(standard=0,adverse=0)

    def act(self,i,feature,previous_action=(0.,0.),slew_limiter=True):
        adverse=toward_load_score(feature)>self.threshold
        self.counts['adverse' if adverse else 'standard']+=1
        policy=self.adverse if adverse else self.standard
        return policy.act(i,feature,previous_action,slew_limiter=slew_limiter)

    @staticmethod
    def to_env_action(cmd):return IndependentActors.to_env_action(cmd)

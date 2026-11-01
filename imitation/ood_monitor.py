"""Training-support indicator for changed wind/current conditions.

Uses only the *training* split of previously recorded demonstrations. The
Euclidean scaled distance is NOT a failure probability, statistical confidence,
or a certified operational domain. It is diagnostic only and has no peer inputs.
"""
from __future__ import annotations
import numpy as np

class EnvironmentalSupport:
    def __init__(self,dataset,agent_id=1,wind_scale_mps=3.,current_scale_mps=.1,warning_distance=3.):
        with np.load(dataset) as data:
            f=data['X'];mask=(data['split']=='train')&(data['agent']==int(agent_id))&(data['t_s']==0)
            init=np.asarray(f[mask],dtype=float)
        if len(init)==0:raise ValueError('No training scenario initial samples')
        self.env_vectors=np.stack([init[:,15]*10.*init[:,16],init[:,15]*10.*init[:,17],
                                   init[:,18]*.5*init[:,19],init[:,18]*.5*init[:,20]],axis=1)
        self.scale=np.array([wind_scale_mps,wind_scale_mps,current_scale_mps,current_scale_mps])
        if not np.all(self.scale>0):raise ValueError('Non-positive scaling')
        self.warning_distance=float(warning_distance)
    def evaluate(self,local_feature):
        x=np.asarray(local_feature,dtype=float)
        if x.ndim!=1 or x.size<21:raise ValueError('Unexpected local feature shape')
        e=np.array([x[15]*10*x[16],x[15]*10*x[17],x[18]*.5*x[19],x[18]*.5*x[20]])
        dist=np.sqrt(np.sum(((self.env_vectors-e)/self.scale)**2,axis=1))
        best=float(np.min(dist))
        return dict(training_context_distance=best,environment_support_warning=best>self.warning_distance)

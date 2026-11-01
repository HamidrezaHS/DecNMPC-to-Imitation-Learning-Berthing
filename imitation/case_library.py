"""Scenario-conditioned nonparametric multi-agent imitation policy.

Pre-recorded TRAIN trajectories are divided into contexts. At evaluation,
all four tugs execute independent neighbor-based policies without calling
NMPC or reading peer states. A local history of the initial task context is
allowed. This is a supervised *retrieval baseline*, not a trained neural actor
and not evidence of universal generalization.
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from scipy.spatial import cKDTree

class ScenarioMemoryActors:
    def __init__(self,dataset,neighbors=3,cases=2,environment_scale=1.,context_temperature=.8):
        with np.load(dataset) as src:
            m=(src['split']=='train')
            X=src['X'][m].astype(np.float64)
            Y=src['Y'][m].astype(np.float64)
            A=src['agent'][m]
            C=src['case'][m]
            T=src['t_s'][m]
        self.neighbors=neighbors;self.n_cases=cases
        self.env_scale=float(environment_scale);self.temp=float(context_temperature)
        self.initial_context=[None]*4
        self.mem=[[] for _ in range(4)]
        # Focus on state/progress and dynamic response instead of previous actions.
        weight=np.array([2.3,3.5,1.1,.25,1.3,1.8,1.0,.12,.12,.08,.08,.04,.04,.03,.08,0.,0.,0.,0.,0.,0.,.25,.02])
        assert weight.size==X.shape[1]
        for agent in range(1,5):
            mask=A==agent
            groups=sorted(set(C[mask].tolist()))
            for case in groups:
                r=(A==agent)&(C==case)
                x=X[r];y=Y[r];t=T[r]
                order=t.argsort();x=x[order];y=y[order];t=t[order]
                first=x[0]
                context=self._context(first)
                state=x*weight
                self.mem[agent-1].append(dict(case=case,context=context,tree=cKDTree(state),y=y,t=t))
        if any(not cases for cases in self.mem):raise ValueError('No training data for some agent')
        self.weight=weight

    @staticmethod
    def _context(x):
        # Wind/current earth-fixed vectors, initial heading, initial ship--target error in body frame.
        W=float(x[15])*10.;C=float(x[18])*.5
        return np.array([W*x[16],W*x[17],C*x[19],C*x[20],
                         math.atan2(x[2],x[3]),x[0]*200.,x[1]*200.])

    def _context_distance(self,a,b):
        diff=a-b
        # Overall source context in actual units; winds in m/s, currents m/s, heading radians.
        return float(np.sqrt(((diff[:2]/2.8)**2).sum()+((diff[2:4]/.09)**2).sum()+
               (diff[4]/math.radians(10.))**2+((diff[5:]/50.)**2).sum()))*self.env_scale

    def act(self,i,feature,previous_action=(0.,0.),slew_limiter=True):
        j=i-1;x=np.asarray(feature,dtype=np.float64)
        if self.initial_context[j] is None:self.initial_context[j]=self._context(x)
        context=self.initial_context[j]
        dist=np.array([self._context_distance(context,z['context']) for z in self.mem[j]])
        top=np.argsort(dist)[:self.n_cases]
        out=[];weights=[]
        for idx in top:
            item=self.mem[j][idx]
            ds,rows=item['tree'].query(x*self.weight,k=self.neighbors)
            ds=np.atleast_1d(ds);rows=np.atleast_1d(rows)
            ws=1/np.maximum(ds,.005)**2
            pred=(item['y'][rows]*ws[:,None]).sum(0)/ws.sum()
            out.append(pred)
            # Weight based on environment closeness and current match.
            weights.append(np.exp(-.5*(dist[idx]/self.temp)**2)/max(.04,ds.min())**.5)
        weights=np.asarray(weights);weights/=max(1e-20,weights.sum())
        y=(np.asarray(out)*weights[:,None]).sum(0)
        f=float(np.clip(y[0],0.,.9 if i<3 else .65));off=float(np.clip(y[1]*20.,-20,20))
        if f<.02:f=0.
        if slew_limiter:
            prev=np.asarray(previous_action,float)
            f=float(np.clip(f,max(0.,prev[0]-.38),prev[0]+.38))
            if prev[0]>=.06:off=float(np.clip(off,prev[1]-24.,prev[1]+24.))
        return np.array([f,off],np.float32)

    @staticmethod
    def to_env_action(command):
        f,off=np.asarray(command,float)
        return np.array([2*f-1.,off/20.],dtype=np.float32)

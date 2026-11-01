import math
import numpy as np
import pytest
from imitation.features import features
from imitation.context_routed import toward_load_score
from imitation.progress_assist import LocalProgressAssist
from imitation.train_paper import sampler_weights,loss_function
import torch

def _feat(wdeg):
    return features([0,0,math.radians(20),0,0,0],[-75,50,0],0,[50,-100,0],
                    8,math.radians(wdeg),.25,math.radians(wdeg))

def test_environment_projection_distinguishes_opposite_forcing():
    assert toward_load_score(_feat(270))>5
    assert toward_load_score(_feat(90)) < -5
    assert abs(toward_load_score(_feat(0))) < 8

def test_local_action_trim_requires_causal_progress_window():
    class Obs:pass
    obs=Obs();obs.ship=np.array([50,-85,0,0,0,0.],float)
    obs.own=np.array([0,0,0.]);obs.own_thrust=0.;obs.time_s=0.
    obs.wind_mps=8.;obs.wind_to_rad=math.radians(270)
    obs.current_mps=.25;obs.current_to_rad=math.radians(270)
    p=LocalProgressAssist(1,[50,-100,0],toward_only=True)
    old=np.array([.05,0.])
    for t in range(0,460,10):
        obs.time_s=float(t);new,dbg=p.apply(obs,old)
    assert new[0]>old[0] and dbg['intensity']>0 and p.count>0
    assert new[0]<=.9

def test_no_trim_for_adverse_gate_in_opposite_flow():
    class Obs:pass
    obs=Obs();obs.ship=np.array([50,-85,0,0,0,0.],float)
    obs.own=np.array([0,0,0.]);obs.own_thrust=0.;obs.time_s=0.
    obs.wind_mps=8.;obs.wind_to_rad=math.radians(90)
    obs.current_mps=.25;obs.current_to_rad=math.radians(90)
    p=LocalProgressAssist(1,[50,-100,0],toward_only=True)
    for t in range(0,460,10):
        obs.time_s=float(t);new,_=p.apply(obs,[.05,0])
    assert new[0]==pytest.approx(.05)
    assert p.count==0

def test_balanced_sampler_does_not_leak_test_rows():
    x=np.zeros((14,23));case=np.array(['A']*4+['B']*6+['C']*4)
    a=np.ones(14,int);split=np.array(['train']*10+['test']*4)
    # agent 2-4 must appear in dataset; construct complete version
    xx=np.tile(x,(4,1));cc=np.tile(case,4);ss=np.tile(split,4)
    aa=np.repeat(np.arange(1,5),14)
    w=sampler_weights(xx,cc,aa,ss)
    assert np.all(w[ss=='test']==0)
    assert np.isclose(w[(cc=='A')&(aa==1)].sum(),1)
    assert np.isclose(w[(cc=='B')&(aa==1)].sum(),1)

def test_physical_loss_is_finite_and_zero_at_target():
    p=torch.tensor([[.5,.5],[0.,.2]],dtype=torch.float32)
    assert float(loss_function(p,p))==pytest.approx(0,abs=1e-12)

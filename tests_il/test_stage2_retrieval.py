"""Stage-2 data isolation and independent policy tests."""
import numpy as np
from imitation.features import N_FEATURES,features
from imitation.retrieval import RetrievalActors
from imitation.case_library import ScenarioMemoryActors


def _fixture(tmp_path):
    rng=np.random.default_rng(17)
    x=rng.normal(size=(64,N_FEATURES)).astype(np.float32)
    y=np.column_stack((rng.uniform(0,.7,64),rng.uniform(-1,1,64))).astype(np.float32)
    agent=np.tile(np.arange(1,5,dtype=np.int32),16)
    split=np.repeat(np.array(['train']*8+['val']*4+['test']*4),4)
    t=np.tile(np.arange(16,dtype=np.float32),4)
    case=np.repeat(np.array(['s0']*8+['s1']*4+['s2']*4),4)
    # Each agent's train data spans 8 samples, so k-neighbor lookup is valid.
    p=tmp_path/'data.npz'
    np.savez_compressed(p,X=x,Y=y,agent=agent,split=split,t_s=t,case=case)
    return p,x,y,agent,split,t,case


def test_retrieval_ignores_validation_and_test_labels(tmp_path):
    p,x,y,a,s,t,c=_fixture(tmp_path)
    original=RetrievalActors(p,neighbors=3)
    probe=x[0].copy();v=original.act(1,probe)
    changed=y.copy();changed[s!='train']=np.array([.9,-1.],np.float32)
    p2=tmp_path/'modified.npz'
    np.savez_compressed(p2,X=x,Y=changed,agent=a,split=s,t_s=t,case=c)
    next=RetrievalActors(p2,neighbors=3)
    assert np.array_equal(v,next.act(1,probe))


def test_retrieval_actor_is_locally_bounded(tmp_path):
    p,x,*_=_fixture(tmp_path)
    policy=RetrievalActors(p,neighbors=3)
    for i in range(1,5):
        act=policy.act(i,x[10+i],previous_action=(0.,0.))
        assert act.shape==(2,)
        assert 0<=act[0]<=min(.38,.9 if i<=2 else .65)
        assert -20<=act[1]<=20
        assert -1<=policy.to_env_action(act)[0]<=1


def test_ship_state_feature_does_not_accept_peer_inputs():
    ship=np.zeros(6); own=np.zeros(3); goal=np.array([50.,-100.,0.])
    vec=features(ship,own,0.,goal,8.,0.,.25,.0)
    assert vec.shape==(N_FEATURES,)
    assert np.isfinite(vec).all()

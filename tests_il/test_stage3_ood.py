import numpy as np
from imitation.ood_monitor import EnvironmentalSupport
from imitation.features import N_FEATURES

def test_monitor_uses_training_scenarios_only(tmp_path):
    X=np.zeros((8,N_FEATURES),np.float32)
    X[:4,15]=.8;X[:4,16]=0;X[:4,17]=1;X[:4,18]=.5;X[:4,19]=0;X[:4,20]=1
    X[4:,15]=.8;X[4:,16]=0;X[4:,17]=-1;X[4:,18]=.5;X[4:,19]=0;X[4:,20]=-1
    splits=np.array(['train']*4+['test']*4)
    t=np.zeros(8); agent=np.tile([1,2,3,4],2)
    p=tmp_path/'d.npz';np.savez_compressed(p,X=X,split=splits,t_s=t,agent=agent)
    monitor=EnvironmentalSupport(p,agent_id=1)
    assert monitor.evaluate(X[0])['training_context_distance']<1e-5
    assert monitor.evaluate(X[4])['environment_support_warning']

def test_monitor_does_not_change_policy_commands(tmp_path):
    X=np.zeros((4,N_FEATURES),np.float32);X[:,15]=.8;X[:,17]=1
    p=tmp_path/'x.npz';np.savez_compressed(p,X=X,split=np.array(['train']*4),t_s=np.zeros(4),agent=np.arange(1,5))
    r=EnvironmentalSupport(p).evaluate(X[0])
    assert set(r)=={'training_context_distance','environment_support_warning'}

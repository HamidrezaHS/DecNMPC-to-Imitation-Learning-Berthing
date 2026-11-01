import numpy as np
from types import SimpleNamespace
from imitation.safety_filter import LocalActionFilter
from imitation.train_robust import training_weights

def fixture(agent=1,heading=0.,speed=-.12,ship_y=-60.):
    hull=np.array([[-160,-29],[160,-29],[160,29],[-160,29]],float)
    side=+1
    anchor=np.array([(-96 if agent in (1,3) else 96),29.])
    f=LocalActionFilter(agent,[50.,-100.,0.],hull,anchor,side)
    # fixed fender face 6m below target low: y = -135
    own=np.array([-96.,-5.,0.]) if agent<3 else np.array([-160.,ship_y+100.,0.])
    sh=np.array([0.,ship_y,heading,0.,speed,0.])
    obs=SimpleNamespace(ship=sh,own=np.r_[own,[0.,0.,0.]])
    return f,obs

def test_push_reduced_when_quay_approach_risky():
    f,o=fixture(1,speed=-.13,ship_y=-85)
    result,diag=f.apply(o,[.6,5.],previous_action=[.6,5.])
    assert 0<=result[0]<.6
    assert diag['intervened'] and diag['risk']>0

def test_tow_raised_when_quay_approach_risky():
    f,o=fixture(3,speed=-.13,ship_y=-85)
    result,diag=f.apply(o,[.1,3.],previous_action=[.1,3.])
    assert result[0]>.1 and diag['intervened']

def test_far_or_slow_does_not_modify_command():
    f,o=fixture(1,speed=-.03,ship_y=-30)
    result,diag=f.apply(o,[.22,-3.],previous_action=[.22,-3.])
    assert np.allclose(result,[.22,-3.]) and not diag['intervened']

def test_slew_is_reapplied_after_filter():
    f,o=fixture(3,speed=-.2,ship_y=-100)
    result,diag=f.apply(o,[.02,0.],previous_action=[0.,0.])
    assert result[0]<=.38 and result[0]>=0

def test_weighted_training_respects_heldout_groups():
    X=np.zeros((16,23),np.float32);agent=np.tile([1,2,3,4],4)
    cases=np.repeat(['a','a','b','c'],4)
    splits=np.repeat(['train','train','val','test'],4)
    w=training_weights(X,cases,agent,splits)
    assert np.all(w[:8]>0) and np.all(w[8:]==0)

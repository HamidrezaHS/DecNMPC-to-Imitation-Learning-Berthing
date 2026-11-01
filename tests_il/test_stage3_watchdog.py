import pytest
from imitation.rollout import simulate

def test_risk_abort_requires_filter(tmp_path):
    with pytest.raises(ValueError,match='requires safety_filter'):
        simulate('missing',tmp_path,risk_abort=True)

def test_bad_wall_time_limit(tmp_path):
    with pytest.raises(ValueError,match='must be positive'):
        simulate('missing',tmp_path,wall_time_limit_s=-1.)

def test_mass_scale_restricted(tmp_path):
    with pytest.raises(ValueError,match='ship_mass_scale'):
        simulate('missing',tmp_path,ship_mass_scale=0.)

def test_sensor_noise_restricted(tmp_path):
    with pytest.raises(ValueError,match='sensor std'):
        simulate('missing',tmp_path,wind_sensor_std_mps=-1.)

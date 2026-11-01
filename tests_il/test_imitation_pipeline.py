from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import pytest
from imitation.dataset import collect_case
from imitation.features import FEATURE_NAMES,from_record,from_observation,N_FEATURES
from imitation.policy import IndependentActors
from decentralized_nmpc.controller import TugObservation

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'expert_archive'/'environment_4x4_v31'/'combined_psi20_w8_wd90_c0p25_cd90'


def rows(path):
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def test_archive_decision_predecision_alignment():
    X,Y,agents,times,name=collect_case(CASE,ROOT/'expert_archive')
    assert len(X)==len(Y)==len(agents)==len(times)
    assert X.shape[1]==N_FEATURES==len(FEATURE_NAMES)
    decisions=rows(CASE/'local_nmpc_decisions.csv');trajectory=rows(CASE/'trajectory.csv')
    assert float(decisions[0]['observation_time_s'])==pytest.approx(float(trajectory[0]['time_s']))
    assert float(decisions[0]['t_s'])==pytest.approx(float(trajectory[1]['time_s']))
    assert np.array_equal(agents[:4],np.arange(1,5))
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y))


def test_feature_matches_sensor_and_ignores_peer_fields():
    rec=rows(CASE/'trajectory.csv')
    cfg=json.loads((CASE/'configuration.json').read_text())
    goal=np.array([cfg['target_x'],cfg['target_y'],0.])
    a=rec[0]
    def construct_obs(r,i):
        sh=np.array([float(r[z]) for z in ['x_m','y_m','psi_deg','u_mps','v_mps','r_deg_s']]);sh[2:3]*=np.pi/180.;sh[5]*=np.pi/180.
        tug=np.array([float(r[f'tug{i}_{z}']) for z in ['x_m','y_m','psi_deg']]);tug[2]*=np.pi/180.
        return TugObservation(ship=sh,own=np.r_[tug,[0.,0.,0.]],
                              own_thrust=float(r[f'tug{i}_thrust_N']),own_azimuth=1e6,
                              time_s=float(r['time_s']),wind_mps=float(r['wind_mps']),
                              wind_to_rad=np.deg2rad(float(r['wind_to_deg'])),
                              current_mps=float(r['current_mps']),
                              current_to_rad=np.deg2rad(float(r['current_to_deg'])))
    obs=construct_obs(a,1)
    f1=from_record(a,None,1,goal)
    f2=from_observation(obs,goal)
    assert np.allclose(f1,f2)
    corrupt=dict(a);corrupt['tug2_x_m']='999999';corrupt['tug4_psi_deg']='-999';corrupt['tug3_fraction']='-99'
    assert np.array_equal(f1,from_record(corrupt,None,1,goal))
    # actuator azimuth and raw own velocity are deliberately excluded from archived 23-feature schema
    obs2=TugObservation(ship=obs.ship,own=np.r_[obs.own[:3],[6.,-9.,10.]],own_thrust=obs.own_thrust,
                        own_azimuth=-9e4,time_s=obs.time_s,
                        wind_mps=obs.wind_mps,wind_to_rad=obs.wind_to_rad,
                        current_mps=obs.current_mps,current_to_rad=obs.current_to_rad)
    assert np.array_equal(f2,from_observation(obs2,goal))


def test_time_difference_uses_only_past_own_state():
    a,b=rows(CASE/'trajectory.csv')[:2]
    goal=np.array([50.,-100.,0.])
    X=from_record(b,a,1,goal)
    assert np.isfinite(X).all()
    changed=dict(b);changed['tug4_y_m']='1e9'
    assert np.array_equal(X,from_record(changed,a,1,goal))
    # no future row is accessed by the constructor


def test_policy_returns_finite_bounded_commands():
    checkpoint=ROOT/'il_results/bc'
    if not checkpoint.is_dir():pytest.skip('no optional pretrained BC checkpoint')
    actors=IndependentActors(checkpoint)
    x=np.zeros(N_FEATURES,dtype=np.float32)
    for i in range(1,5):
        cmd=actors.act(i,x)
        assert 0<=cmd[0]<= (.9 if i<3 else .65)
        assert abs(cmd[1])<=20.
        a=actors.to_env_action(cmd)
        assert a.shape==(2,) and np.isfinite(a).all()

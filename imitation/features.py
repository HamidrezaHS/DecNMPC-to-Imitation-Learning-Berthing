"""Causally available features for one tug at each *pre-action* decision instant.

The archived ship-state teacher records do not contain tug body velocities
or applied azimuth. Those are NOT fabricated. Instead, the actor uses only:
ship pose/velocity, its own pose/thrust and the immediately preceding *own*
pose and command, mission pose, and local wind/current measurements.
The finite-difference own velocity is available without future information.
No other tug state or planned action is used.
"""
from __future__ import annotations
import math
import numpy as np

FEATURE_NAMES = (
    'goal_body_x_200','goal_body_y_200','sin_heading_error','cos_heading_error',
    'ship_u_0p3','ship_v_0p3','ship_r_deg_0p3',
    'own_rel_ship_x_200','own_rel_ship_y_100',
    'sin_rel_heading','cos_rel_heading',
    'own_fd_u_0p6','own_fd_v_0p6','own_fd_r_deg_1',
    'own_thrust_over_max','wind_mps_10','wind_toward_cos','wind_toward_sin',
    'current_mps_0p5','current_toward_cos','current_toward_sin',
    'previous_fraction','previous_offset_20deg'
)
N_FEATURES=len(FEATURE_NAMES)
MAX_FORCE_N=100.0*9806.65   # matches the physical simulation configuration


def wrap(a:float)->float:
    return (float(a)+math.pi)%(2*math.pi)-math.pi


def features(ship,own,own_thrust,goal,wind_mps,wind_to_rad,current_mps,current_to_rad,
             previous_action=(0.,0.),previous_own=None,previous_time=None,time_s=0.):
    """Return one local actor observation. previous_action=(fraction, offset degrees)."""
    sh=np.asarray(ship,dtype=np.float64);o=np.asarray(own,dtype=np.float64)
    goal=np.asarray(goal,dtype=np.float64)
    if sh.shape!=(6,) or o.shape!=(3,) or goal.shape!=(3,):
        raise ValueError('Expected ship(6), own pose(3), goal(3)')
    c,s=math.cos(sh[2]),math.sin(sh[2]);R_T=np.array([[c,s],[-s,c]])
    e_goal=R_T@(goal[:2]-sh[:2]);rel=R_T@(o[:2]-sh[:2])
    prev=np.asarray(previous_action,dtype=np.float64)
    if prev.shape!=(2,):raise ValueError('previous_action must have 2 physical values')
    if previous_own is not None and previous_time is not None and time_s>previous_time+1e-7:
        previous_own=np.asarray(previous_own,dtype=np.float64)
        dt=time_s-previous_time
        # Translation measured over previous decision interval in CURRENT own body frame.
        owc,ows=math.cos(o[2]),math.sin(o[2])
        v_world=(o[:2]-previous_own[:2])/dt
        u_fd=owc*v_world[0]+ows*v_world[1]
        v_fd=-ows*v_world[0]+owc*v_world[1]
        r_fd=wrap(o[2]-previous_own[2])/dt
    else:u_fd=v_fd=r_fd=0.
    w=float(wind_to_rad);cur=float(current_to_rad)
    vec=np.array([
        e_goal[0]/200.,e_goal[1]/200.,math.sin(wrap(sh[2]-goal[2])),math.cos(wrap(sh[2]-goal[2])),
        sh[3]/.3,sh[4]/.3,sh[5]/math.radians(.3),
        rel[0]/200.,rel[1]/100.,math.sin(wrap(o[2]-sh[2])),math.cos(wrap(o[2]-sh[2])),
        u_fd/.6,v_fd/.6,r_fd/math.radians(1.),
        float(own_thrust)/MAX_FORCE_N,float(wind_mps)/10.,math.cos(w),math.sin(w),
        float(current_mps)/.5,math.cos(cur),math.sin(cur),
        prev[0],prev[1]/20.,
    ],dtype=np.float32)
    if vec.shape!=(N_FEATURES,) or not np.all(np.isfinite(vec)):
        raise ValueError('Non-finite or incorrect local feature vector')
    return np.clip(vec,-20.,20.)


def from_record(row,previous_row,agent,goal):
    """Build from stored telemetry BEFORE the paired teacher decision."""
    i=int(agent)
    if not 1<=i<=4:raise ValueError('agent must be 1..4')
    time_s=float(row['time_s'])
    own=np.array([float(row[f'tug{i}_{p}']) for p in ('x_m','y_m','psi_deg')],dtype=float)
    own[2]=math.radians(own[2])
    previous_own=None
    if previous_row is not None:
        previous_own=np.array([float(previous_row[f'tug{i}_{p}']) for p in ('x_m','y_m','psi_deg')],dtype=float)
        previous_own[2]=math.radians(previous_own[2])
    sh=np.array([float(row[z]) for z in ['x_m','y_m','psi_deg','u_mps','v_mps','r_deg_s']])
    sh[2]=math.radians(sh[2]);sh[5]=math.radians(sh[5])
    f=float(row[f'tug{i}_fraction']) if row[f'tug{i}_fraction'] else 0.
    off=float(row[f'tug{i}_offset_deg']) if row[f'tug{i}_offset_deg'] else 0.
    return features(sh,own,float(row[f'tug{i}_thrust_N']),goal,
        float(row['wind_mps']),math.radians(float(row['wind_to_deg'])),
        float(row['current_mps']),math.radians(float(row['current_to_deg'])),
        previous_action=(f,off),previous_own=previous_own,
        previous_time=float(previous_row['time_s']) if previous_row is not None else None,
        time_s=time_s)


def from_observation(obs,goal,previous_action=(0.,0.),previous_own=None,previous_time=None):
    return features(obs.ship,obs.own[:3],obs.own_thrust,goal,obs.wind_mps,obs.wind_to_rad,
        obs.current_mps,obs.current_to_rad,previous_action,previous_own,previous_time,obs.time_s)

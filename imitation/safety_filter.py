"""Local no-expert action filter and early-warning safety diagnostics.

This is a *heuristic engineering filter*, NOT a control barrier function, a
formal viability guarantee, or a certificate of collision-free navigation.
Every tug uses only shared ship pose/velocity, own tug pose and its own action;
there are no peer-tug measurements or future peer actions. Geometry and berth
line are public static mission information.
"""
from __future__ import annotations
import math
import numpy as np
from .features import MAX_FORCE_N


def rot(psi):
    c,s=math.cos(float(psi)),math.sin(float(psi))
    return np.array([[c,-s],[s,c]],dtype=float)


class LocalActionFilter:
    def __init__(self,agent_id,goal,hull_points,attachment,side,
                 lookahead_gap_m=95.,soft_closing_mps=.078,
                 max_tow_rescue_fraction=.48):
        self.agent_id=int(agent_id)
        if not 1<=self.agent_id<=4:raise ValueError('agent_id 1..4')
        self.goal=np.asarray(goal,float).reshape(3)
        self.hull=np.asarray(hull_points,float)
        if self.hull.ndim!=2 or self.hull.shape[1]!=2:raise ValueError('Hull shape Nx2')
        self.attachment=np.asarray(attachment,float).reshape(2)
        self.side=int(side)
        if self.side not in (-1,1):raise ValueError('side should be -1 or +1')
        target_low=(self.hull@rot(self.goal[2]).T+self.goal[:2])[:,1].min()
        self.quay_y=float(target_low-6.)
        self.lookahead=float(lookahead_gap_m)
        self.soft_closing=float(soft_closing_mps)
        self.rescue_fraction=float(max_tow_rescue_fraction)
        self.count=0;self.max_risk=0.;self.min_tow_side_margin=1e9

    def apply(self,obs,command,previous_action=(0.,0.)):
        """Return filtered [fraction, offset_deg], diagnostic dictionary.

        No plant object, simulator future or peer observations are accessed.
        Due to dynamics/coupling this filter cannot guarantee safety.
        """
        sh=np.asarray(obs.ship,dtype=float)
        own=np.asarray(obs.own[:3],dtype=float)
        a=np.asarray(command,dtype=float).copy()
        prev=np.asarray(previous_action,dtype=float)
        if sh.shape!=(6,) or own.shape!=(3,) or a.shape!=(2,) or not np.isfinite(a).all():
            raise ValueError('Malformed local safety input')
        R=rot(sh[2]);world_vel=R@sh[3:5]
        gap=float((self.hull@R.T+sh[:2])[:,1].min()-self.quay_y)
        closing=float(max(0.,-world_vel[1]))
        # Earlier than the plant's gap<45 m & closing>0.095 m/s unsafe gate.
        # Dimensionless risk score is a HEURISTIC, not a probability.
        proximity=float(np.clip((self.lookahead-gap)/45.,0.,1.))
        velocity=float(np.clip((closing-self.soft_closing)/.042,0.,1.))
        risk=float(proximity*velocity)
        self.max_risk=max(self.max_risk,risk)
        margin=float((own[:2]-(sh[:2]+R@self.attachment))@(R@np.array([0.,self.side])))
        self.min_tow_side_margin=min(self.min_tow_side_margin,margin) if self.agent_id>=3 else self.min_tow_side_margin
        before=a.copy()
        reasons=[]
        if risk>0:
            if self.agent_id<=2:
                # Compression on the free-side pushes TOWARD the quay; attenuate.
                a[0]*=(1.-.95*risk)
            else:
                # Astern tow pulls toward the free side. Its physical line must
                # already be taut to influence the tanker; no force guarantee.
                a[0]+=risk*max(0.,self.rescue_fraction-a[0])
            reasons.append('quay_approach_speed')
        if self.agent_id>=3 and margin<18.:
            # If own tug approaches the wrong shell side, request astern pull;
            # contact/rope physics remains with the original plant.
            danger=float(np.clip((18.-margin)/15.,0.,1.))
            a[0]+=danger*max(0.,.42-a[0])
            reasons.append('tow_side_margin')
        a[0]=float(np.clip(a[0],0.,.9 if self.agent_id<=2 else .65))
        a[1]=float(np.clip(a[1],-20.,20.))
        # Physical command slew bound is applied after filter as well.
        a[0]=float(np.clip(a[0],max(0.,prev[0]-.38),prev[0]+.38))
        if prev[0]>=.06:a[1]=float(np.clip(a[1],prev[1]-24,prev[1]+24))
        modified=bool(np.max(np.abs(before-a))>1.e-7)
        if modified:self.count+=1
        return a.astype(np.float32),dict(intervened=modified,risk=risk,quay_gap_m=gap,
                       closing_speed_mps=closing,tow_side_margin_m=margin,
                       reason='+'.join(reasons) if modified else '')

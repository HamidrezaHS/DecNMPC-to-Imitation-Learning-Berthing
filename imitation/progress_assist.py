"""Optional LOCAL progress feedback for learned policies.

A simple slow-moving normal-direction trim for independent push/tow tugs.
This is a heuristic mixed learned/feedback controller, NOT pure end-to-end IL,
not a control barrier, and not a guarantee of safe berthing. It uses ONLY the
shared measured ship state and the tug's local action, no peer communication.
It exists as an isolated ablation, not silently added to the neural baseline.
"""
from __future__ import annotations
import math
import numpy as np

class LocalProgressAssist:
    def __init__(self,agent,goal,max_boost_push=.18,max_relief_tow=.17,
                 window_s=110.,rise_s=280.,toward_only=False,threshold=2.):
        if int(agent) not in (1,2,3,4):raise ValueError('agent must be 1..4')
        self.agent=int(agent);self.goal=np.asarray(goal,float).reshape(3)
        self.normal=np.array([-math.sin(self.goal[2]),math.cos(self.goal[2])])
        self.direction=None;self.stall_s=0.;self.prev_time=None
        self.max_push=float(max_boost_push);self.max_tow=float(max_relief_tow)
        self.window=float(window_s);self.rise=float(rise_s);self.count=0
        self.max_trim=0.
        self.toward_only=bool(toward_only);self.threshold=float(threshold)

    def apply(self,obs,command):
        ship=np.asarray(obs.ship,float)
        if ship.shape!=(6,):raise ValueError('ship must have six states')
        if self.direction is None:
            sign=np.sign((self.goal[:2]-ship[:2])@self.normal)
            self.direction=self.normal*(1. if sign>=0 else -1.)
        delta=self.goal[:2]-ship[:2]
        distance=float(delta@self.direction)
        along=float(delta@np.array([self.direction[1],-self.direction[0]]))
        err_heading=abs((float(ship[2])-float(self.goal[2])+math.pi)%(2*math.pi)-math.pi)
        c,s=math.cos(ship[2]),math.sin(ship[2])
        world_vel=np.array([c*ship[3]-s*ship[4],s*ship[3]+c*ship[4]])
        closing=float(world_vel@self.direction)
        dt=0. if self.prev_time is None else max(0.,min(15.,float(obs.time_s)-self.prev_time))
        self.prev_time=float(obs.time_s)
        # Progressive speed request, decreasing near the target: no thrust boost after arrival.
        reference=min(.034,math.sqrt(max(0.,2.*.000045*distance)))
        gated=bool(0.7<distance<34. and abs(along)<8. and err_heading<math.radians(2.8))
        if self.toward_only:
            from .context_routed import toward_load_score
            from .features import from_observation
            # Use only directly measured ship + weather data, no peer states.
            # Construct initial-free feature; gate needs ONLY mission/heading/weather fields.
            feat=from_observation(obs,self.goal)
            gated=gated and toward_load_score(feat)>self.threshold
        stalled=gated and closing<max(.006,reference*.58)
        if stalled:self.stall_s+=dt
        else:self.stall_s=max(0.,self.stall_s-2.5*dt)
        # Gently taper rather than abruptly switching off four meters before acceptance.
        taper=float(np.clip((distance-.65)/3.5,0.,1.))
        intensity=float(np.clip((self.stall_s-self.window)/self.rise,0.,1.))*taper if gated else 0.
        action=np.asarray(command,float).copy()
        old=action.copy()
        if self.agent<=2:action[0]=min(.90,action[0]+intensity*self.max_push)
        else:action[0]=max(0.,action[0]-intensity*self.max_tow)
        changed=bool(abs(action[0]-old[0])>1e-8)
        if changed:self.count+=1
        self.max_trim=max(self.max_trim,abs(action[0]-old[0]))
        return action.astype(np.float32),dict(intervened=changed,intensity=intensity,
            stall_duration_s=self.stall_s,reference_speed_mps=reference,
            measured_closing_mps=closing,normal_distance_m=distance,
            along_distance_m=along)

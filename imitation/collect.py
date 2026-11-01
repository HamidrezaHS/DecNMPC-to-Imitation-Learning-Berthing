"""Run the unchanged ship-state NMPC expert to record extra demonstrations.

Stores new trajectories under expert_archive/user_generated; any newly
recorded case is treated as training data by imitation.dataset. Do NOT use the
held-out evaluation scenarios for collecting new training labels.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from run_nmpc import rollout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='additional_expert_case')
    p.add_argument('--out',default='expert_archive/user_generated')
    p.add_argument('--psi0',type=float,default=20.)
    p.add_argument('--wind-mps',type=float,default=8.)
    p.add_argument('--wind-to-deg',type=float,default=90.)
    p.add_argument('--current-mps',type=float,default=.25)
    p.add_argument('--current-to-deg',type=float,default=90.)
    p.add_argument('--start-x',type=float,default=0.)
    p.add_argument('--start-y',type=float,default=0.)
    p.add_argument('--target-x',type=float,default=50.)
    p.add_argument('--target-y',type=float,default=-100.)
    p.add_argument('--duration',type=float,default=6500.)
    p.add_argument('--maxiter',type=int,default=6)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--render',action='store_true')
    a=p.parse_args()
    if '/' in a.name or '\\' in a.name or a.name in ('.','..'):
        p.error('Provide a simple directory name without slashes')
    dest=Path(a.out)/a.name
    if dest.exists():p.error('Output folder exists; use a new --name to preserve archived results')
    m=rollout(out=str(dest),controller_method='ship_state_local',
            psi0=a.psi0,wind_mps=a.wind_mps,wind_to_deg=a.wind_to_deg,
            current_mps=a.current_mps,current_to_deg=a.current_to_deg,
            target_x=a.target_x,target_y=a.target_y,start_x=a.start_x,start_y=a.start_y,
            max_seconds=a.duration,maxiter=a.maxiter,seed=a.seed,render=a.render,
            export_ship_state_demos=True,save_every_seconds=300.,verbose=True)
    if not m['success'] or m['unsafe']:
        print('WARNING: this teacher episode was not safely successful and the dataset builder will reject it.')
if __name__=='__main__':main()

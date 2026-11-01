"""Safe teacher data acquisition from predeclared training-only tasks.

Never train on the predefined noncardinal held-out cases. Failed expert
rollouts are recorded as failures and not accepted as imitation demonstrations.
The solver may take substantial CPU time; each case is stored independently.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from run_nmpc import rollout

TRAIN_ONLY={
    'calm_heading10': dict(psi0=10.,wind_mps=0.,current_mps=0.),
    'calm_heading25': dict(psi0=25.,wind_mps=0.,current_mps=0.),
    'wind45_current90': dict(psi0=20.,wind_mps=8.,wind_to_deg=45.,current_mps=.25,current_to_deg=90.),
    'wind135_current180': dict(psi0=20.,wind_mps=8.,wind_to_deg=135.,current_mps=.25,current_to_deg=180.),
    'wind315_current270': dict(psi0=20.,wind_mps=8.,wind_to_deg=315.,current_mps=.25,current_to_deg=270.),
    'wind270_current315': dict(psi0=20.,wind_mps=8.,wind_to_deg=270.,current_mps=.25,current_to_deg=315.),
    'stronger180_270': dict(psi0=30.,wind_mps=9.,wind_to_deg=180.,current_mps=.3,current_to_deg=270.),
    'weaker90_90':dict(psi0=20.,wind_mps=6.,wind_to_deg=90.,current_mps=.18,current_to_deg=90.),
}
TEST_ONLY={'noncardinal225','cross45_135','counterflow270','heading40'}

def collect(out,tasks,max_seconds=6500.,maxiter=6,seed=123):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in tasks:
        if name not in TRAIN_ONLY:raise ValueError('Only predeclared training tasks allowed')
        p=out/name
        p.mkdir(parents=True,exist_ok=True)
        if (p/'metrics.json').is_file():
            metric=json.loads((p/'metrics.json').read_text());print('[CACHED]',name)
        else:
            print('[TEACHER]',name,flush=True)
            metric=rollout(out=str(p),controller_method='ship_state_local',
                max_seconds=max_seconds,maxiter=maxiter,seed=seed,render=False,
                export_ship_state_demos=True,save_every_seconds=300.,verbose=False,
                target_x=50.,target_y=-100.,**TRAIN_ONLY[name])
        valid=bool(metric.get('success') and not metric.get('unsafe'))
        rows.append(dict(task=name,success=valid,status=metric.get('status','UNKNOWN'),
                         seconds=metric.get('seconds'),path=str(p)))
        (out/'curriculum_status.json').write_text(json.dumps(rows,indent=2))
        if not valid:print('WARNING: not a safely successful teacher demo:',name,flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='expert_archive/stage3_curriculum')
    p.add_argument('--tasks',nargs='+',choices=list(TRAIN_ONLY),default=list(TRAIN_ONLY))
    p.add_argument('--duration',type=float,default=6500.)
    p.add_argument('--maxiter',type=int,default=6);p.add_argument('--seed',type=int,default=123)
    a=p.parse_args();print(json.dumps(collect(a.out,a.tasks,a.duration,a.maxiter,a.seed),indent=2))
if __name__=='__main__':main()

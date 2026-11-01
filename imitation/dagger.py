"""Iterative DAgger on *training* scenarios only, with per-actor expert queries.

Each rollout visits states with a beta-mixture student/teacher controller.  The
expert labels every visited state, and data are appended to the training split
only. The original validation/test scenario splits are left unchanged.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .rollout import simulate
from .train import run_train

TRAIN_TASKS={
 'calm20':dict(psi0=20,wind_mps=0.,current_mps=0.),
 'calm40':dict(psi0=40,wind_mps=0.,current_mps=0.),
 'wind20':dict(psi0=20,wind_mps=8.,wind_to_deg=90.,current_mps=0.),
 'current20':dict(psi0=20,wind_mps=0.,current_mps=.25,current_to_deg=90.),
 'combined20':dict(psi0=20,wind_mps=8.,wind_to_deg=90.,current_mps=.25,current_to_deg=90.),
 'combined180':dict(psi0=20,wind_mps=8.,wind_to_deg=180.,current_mps=.25,current_to_deg=90.),
}


def merge_dagger(base_path,rollout_files,out_path,round_id=1):
    arrays=np.load(base_path)
    X=[arrays['X']];Y=[arrays['Y']];agent=[arrays['agent']];times=[arrays['t_s']]
    groups=[arrays['case']];split=[arrays['split']]
    for k,f in enumerate(rollout_files):
        with np.load(f) as r:
            x=r['X'];y=r['Y'];a=r['agent'];t=r['t_s']
        if len(x)==0:continue
        if x.shape[1]!=X[0].shape[1] or y.shape[1]!=2:raise ValueError('DAgger shape mismatch')
        X.append(x);Y.append(y);agent.append(a);times.append(t)
        groups.append(np.repeat(f'dagger/round_{round_id:02d}/rollout_{k:02d}',len(x)))
        split.append(np.repeat('train',len(x)))
    out_path=Path(out_path);out_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_path,X=np.concatenate(X),Y=np.concatenate(Y),agent=np.concatenate(agent),
                        t_s=np.concatenate(times),case=np.concatenate(groups),split=np.concatenate(split))
    merged=np.load(out_path)
    # Guarantee DAgger never touches the original validation/test distribution.
    for s in ['val','test']:
        initial=arrays['split']==s;current=merged['split']==s
        assert np.array_equal(arrays['X'][initial],merged['X'][current])
        assert np.array_equal(arrays['Y'][initial],merged['Y'][current])
    return out_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='il_results/expert_dataset.npz')
    p.add_argument('--actors',default='il_results/bc')
    p.add_argument('--out',default='il_results/dagger')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--rollouts-per-round',type=int,default=3)
    p.add_argument('--duration',type=float,default=4500.)
    p.add_argument('--epochs',type=int,default=85)
    p.add_argument('--beta-start',type=float,default=.6)
    p.add_argument('--beta-decay',type=float,default=.5)
    p.add_argument('--maxiter',type=int,default=6)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    p.add_argument('--seed',type=int,default=323)
    p.add_argument('--tasks',nargs='+',choices=list(TRAIN_TASKS),default=['combined20','wind20','current20','calm20'])
    a=p.parse_args()
    if a.rounds<1 or a.rollouts_per_round<1:raise ValueError('rounds and rollouts must be positive')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    current_dataset=Path(a.dataset);current_actors=Path(a.actors)
    histories=[]
    for r in range(1,a.rounds+1):
        beta=max(0.,min(1.,a.beta_start*(a.beta_decay**(r-1))))
        runfiles=[]
        for k in range(a.rollouts_per_round):
            name=a.tasks[((r-1)*a.rollouts_per_round+k)%len(a.tasks)]
            d=out/f'round_{r:02d}'/f'{k+1:02d}_{name}'
            print(f'[DAGGER] round {r} rollout {k+1}/{a.rollouts_per_round} {name}, beta={beta}',flush=True)
            metrics=simulate(current_actors,d,max_seconds=a.duration,seed=a.seed+r*100+k,
                             beta=beta,query_expert=True,teacher_maxiter=a.maxiter,
                             verbose=False,device='cpu',**TRAIN_TASKS[name])
            histories.append(dict(round=r,case=name,beta=beta,status=metrics['status'],
                                  success=metrics['success'],unsafe=metrics['unsafe'],
                                  position_error_m=metrics['position_error_m'],
                                  expert_labels=metrics['expert_labels_collected']))
            runfiles.append(d/'dagger_samples.npz')
            (out/'dagger_progress.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
        next_dataset=out/f'aggregated_round_{r:02d}.npz'
        merge_dagger(current_dataset,runfiles,next_dataset,round_id=r)
        next_actor_dir=out/f'actors_round_{r:02d}'
        run_train(next_dataset,next_actor_dir,epochs=a.epochs,device=a.device,
                  seed=a.seed+r)
        current_dataset=next_dataset;current_actors=next_actor_dir
        print(f'[DAGGER] saved {current_actors}',flush=True)
    (out/'dagger_summary.json').write_text(json.dumps(dict(rollouts=histories,
         final_actor_directory=str(current_actors),final_dataset=str(current_dataset)),indent=2),encoding='utf-8')
    print(f'DONE: independent DAgger actors in {current_actors}')
if __name__=='__main__':main()

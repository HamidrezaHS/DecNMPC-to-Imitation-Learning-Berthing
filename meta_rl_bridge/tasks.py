"""Future adaptation task manifest; NOT a meta-reinforcement-learning algorithm.

Creates disjoint evaluation metadata for later episode-wise adaptation work.
Task parameters are privileged evaluator-only labels and never policy inputs.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def generate(seed=2026,n_train=64,n_val=16,n_ood=24):
    if min(n_train,n_val,n_ood)<1:raise ValueError('Positive task counts needed')
    rng=np.random.default_rng(seed)
    specs=[]
    profiles=[('train',n_train,(6.,9.),(.17,.28),(15.,32.),(.92,1.08)),
              ('val',n_val,(6.,9.),(.17,.28),(15.,32.),(.92,1.08)),
              ('ood',n_ood,(9.2,11.),(.29,.38),(34.,45.),(1.10,1.22))]
    for group,n,w,c,heading,mass in profiles:
        for i in range(n):
            specs.append(dict(task_id=f'{group}_{i:04d}',partition=group,seed=int(rng.integers(1,2**31)),
                wind_mps=round(float(rng.uniform(*w)),4),current_mps=round(float(rng.uniform(*c)),5),
                wind_to_deg=round(float(rng.uniform(0,360)),3),
                current_to_deg=round(float(rng.uniform(0,360)),3),
                psi0=round(float(rng.uniform(*heading)),3),ship_mass_scale=round(float(rng.uniform(*mass)),5),
                target_x=50.,target_y=-100.,
                teacher_model_access=False,
                annotation='Ground-truth task values FOR EVALUATOR ONLY. Not observable to local policy.'))
    return dict(schema='FUTURE-META-TASKS-1',status='NOT_TRAINED_NOT_EVALUATED',
                seed=seed,task_count=len(specs),tasks=specs,
                warning='Not a Meta-RL method. Train/val ranges overlap by design; OOD ranges are disjoint. Every episode must limit observations to shared ship + own tug + local measured environment.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='experiments/future_meta_tasks.json');p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--train-count',type=int,default=64);p.add_argument('--val-count',type=int,default=16);p.add_argument('--ood-count',type=int,default=24)
    a=p.parse_args();obj=generate(a.seed,a.train_count,a.val_count,a.ood_count)
    f=Path(a.out);f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(obj,indent=2),encoding='utf8')
    print('Wrote',f,'(no model training or validation executed)')
if __name__=='__main__':main()

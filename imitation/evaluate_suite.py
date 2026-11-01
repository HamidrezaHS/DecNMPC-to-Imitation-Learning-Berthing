"""Reproducible closed-loop evaluation suite for independent IL actors.

No NMPC expert is queried. Safe arrival uses the original 5-second terminal
acceptance monitor; invalid, unsafe, timeout and success remain distinct.

The named scenarios are convenience presets, not statistically sufficient
validation. They include both training-like and outside-training tasks.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from .rollout import simulate

SCENARIOS={
    'main90':dict(psi0=20.,wind_mps=8.,wind_to_deg=90.,current_mps=.25,current_to_deg=90.),
    'heading22':dict(psi0=22.,wind_mps=8.,wind_to_deg=90.,current_mps=.25,current_to_deg=90.),
    'heading24':dict(psi0=24.,wind_mps=8.,wind_to_deg=90.,current_mps=.25,current_to_deg=90.),
    'magnitude_perturbed':dict(psi0=20.,wind_mps=7.7,wind_to_deg=90.,current_mps=.22,current_to_deg=90.),
    'calm20':dict(psi0=20.,wind_mps=0.,current_mps=0.),
    'heldout270':dict(psi0=20.,wind_mps=8.,wind_to_deg=270.,current_mps=.25,current_to_deg=270.),
    'heading25':dict(psi0=25.,wind_mps=8.,wind_to_deg=90.,current_mps=.25,current_to_deg=90.),
    'magnitude_new':dict(psi0=20.,wind_mps=7.9,wind_to_deg=90.,current_mps=.23,current_to_deg=90.),
}

def evaluate(actors,out,tasks,max_seconds=6500,render=False,seed=123):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in tasks:
        if name not in SCENARIOS:raise ValueError(f'Unknown task {name}')
        print(f'[EVALUATE] {name}',flush=True)
        m=simulate(actors,out/name,max_seconds=max_seconds,seed=seed,render=render,verbose=False,
                    beta=0.,query_expert=False,**SCENARIOS[name])
        rows.append(dict(task=name,status=m['status'],success=m['success'],unsafe=m['unsafe'],
            seconds=m['seconds'],position_error_m=m['position_error_m'],
            heading_error_deg=m['heading_error_deg'],
            min_hull_gap_m=m['min_logged_hull_gap_m'],
            mean_inference_ms=m['mean_actor_inference_ms']))
        (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actors',default='il_results/stage2/dagger_balanced/actors_round_02')
    p.add_argument('--out',default='il_results/stage2/suite')
    p.add_argument('--tasks',nargs='+',choices=sorted(SCENARIOS),default=list(SCENARIOS))
    p.add_argument('--duration',type=float,default=6500)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--render',action='store_true')
    a=p.parse_args();evaluate(a.actors,a.out,a.tasks,max_seconds=a.duration,render=a.render,seed=a.seed)
if __name__=='__main__':main()

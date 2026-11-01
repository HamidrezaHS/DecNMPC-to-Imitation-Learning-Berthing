"""Seeded and reproducible safety/generalization suite for student-only policies.

Run the frozen declared scenarios and save a per-attempt record. No expert
queries. All statuses other than SUCCESS are counted as mission failures.
The initial acceptance envelope is NOT sustained berthing/station keeping.
"""
from __future__ import annotations
import argparse,csv,json,sys,subprocess
from pathlib import Path
import numpy as np
from .rollout import simulate

SCENARIOS={
 'trained90':dict(psi0=20.,wind_mps=8.,wind_to_deg=90.,current_mps=.25,current_to_deg=90.),
 'heading25':dict(psi0=25.,wind_mps=8.,wind_to_deg=90.,current_mps=.25,current_to_deg=90.),
 'magnitude79':dict(psi0=20.,wind_mps=7.9,wind_to_deg=90.,current_mps=.23,current_to_deg=90.),
 'calm20':dict(psi0=20.,wind_mps=0.,wind_to_deg=0.,current_mps=0.,current_to_deg=0.),
 'counterflow270':dict(psi0=20.,wind_mps=8.,wind_to_deg=270.,current_mps=.25,current_to_deg=270.),
 'noncardinal225':dict(psi0=20.,wind_mps=8.,wind_to_deg=225.,current_mps=.25,current_to_deg=225.),
 'cross45_135':dict(psi0=20.,wind_mps=8.,wind_to_deg=45.,current_mps=.25,current_to_deg=135.),
 'heading40':dict(psi0=40.,wind_mps=8.,wind_to_deg=90.,current_mps=.25,current_to_deg=90.),
}

def evaluate(actors,out,tasks,seeds=(123,),duration=6500.,safety=False,abort=False,wall_limit=90.,ood_dataset=None,render=False,hard_timeout=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if abort and not safety:raise ValueError('--abort requires --safety-filter')
    results=[]
    for task in tasks:
        if task not in SCENARIOS:raise ValueError(f'Unknown scenario: {task}')
        for seed in seeds:
            folder=out/task/f'seed_{seed}'
            manifest=folder/'metrics.json'
            if manifest.is_file():
                metric=json.loads(manifest.read_text())
                print(f'[CACHED] {task}/{seed}: {metric["status"]}',flush=True)
            else:
                print(f'[RUN] {task}/{seed}',flush=True)
                if hard_timeout is not None:
                    folder.mkdir(parents=True,exist_ok=True)
                    params=SCENARIOS[task]
                    cmd=[sys.executable,'-m','imitation.rollout',
                         '--actors',str(actors),'--out',str(folder),
                         '--psi0',str(params['psi0']), '--wind-mps',str(params['wind_mps']),
                         '--wind-to-deg',str(params['wind_to_deg']),
                         '--current-mps',str(params['current_mps']),
                         '--current-to-deg',str(params['current_to_deg']),
                         '--seed',str(seed),'--duration',str(duration),
                         '--wall-time-limit-s',str(wall_limit),'--quiet']
                    if safety:cmd.append('--safety-filter')
                    if abort:cmd.append('--risk-abort')
                    if render:cmd.append('--render')
                    if ood_dataset:cmd.extend(['--ood-dataset',str(ood_dataset)])
                    try:
                        run=subprocess.run(cmd,capture_output=True,text=True,timeout=hard_timeout)
                        (folder/'process_run.log').write_text(run.stdout+'\nSTDERR:\n'+run.stderr)
                        if run.returncode!=0:raise RuntimeError(f'Rollout exit {run.returncode}. See {folder}/process_run.log')
                        metric=json.loads(manifest.read_text())
                    except subprocess.TimeoutExpired:
                        # The child may stall inside a single simulator substep;
                        # parent process watchdog terminates it without inventing results.
                        metric={'status':'PROCESS_TIMEOUT','success':False,'unsafe':None,
                                'seconds':None,'position_error_m':None,
                                'cpu_elapsed_s':float(hard_timeout),
                                'safety_filter_interventions':None,'environment_support':None}
                        (folder/'process_timeout.json').write_text(json.dumps(metric,indent=2))
                else:
                    metric=simulate(actors,folder,max_seconds=duration,seed=seed,render=render,
                        verbose=False,query_expert=False,beta=0.,safety_filter=safety,risk_abort=abort,
                        ood_dataset=ood_dataset,wall_time_limit_s=wall_limit,**SCENARIOS[task])
            results.append(dict(task=task,seed=seed,status=metric['status'],
                success=bool(metric['status']=='SUCCESS' and metric['success'] and not metric['unsafe']),
                unsafe=(bool(metric['unsafe']) if metric['unsafe'] is not None else None),seconds=metric['seconds'],
                position_error_m=metric['position_error_m'],
                interventions=metric.get('safety_filter_interventions',0),
                environment_support_warning=bool((metric.get('environment_support') or [{}])[0].get('environment_support_warning',False)),
                wallclock_s=metric['cpu_elapsed_s']))
            (out/'summary.json').write_text(json.dumps(results,indent=2))
            with (out/'summary.csv').open('w',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=results[0].keys());wr.writeheader();wr.writerows(results)
    report=dict(completed=len(results),successes=sum(int(x['success']) for x in results),
        unsafe_terminations=sum(x['unsafe'] is True for x in results),
        risk_aborts=sum(x['status']=='RISK_ABORT' for x in results),
        walltime_aborts=sum(x['status']=='WALL_TIME_ABORT' for x in results),
        process_timeouts=sum(x['status']=='PROCESS_TIMEOUT' for x in results),
        success_fraction=sum(int(x['success']) for x in results)/max(1,len(results)),
        warning='Descriptive sample fraction only. No confidence interval or formal safety claim.')
    (out/'score.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actors',default='il_results/stage3/actors_balanced')
    p.add_argument('--out',default='il_results/stage3/eval_protocol')
    p.add_argument('--tasks',nargs='+',choices=sorted(SCENARIOS),default=list(SCENARIOS))
    p.add_argument('--seeds',nargs='+',type=int,default=[123,456,789])
    p.add_argument('--duration',type=float,default=6500.)
    p.add_argument('--wall-time-limit-s',type=float,default=90.)
    p.add_argument('--hard-timeout-s',type=float,default=105.,help='OS process kill timeout, separate from simulation-time guard')
    p.add_argument('--safety-filter',action='store_true')
    p.add_argument('--risk-abort',action='store_true')
    p.add_argument('--ood-dataset',default='il_results/stage2/dagger_balanced/aggregated_round_02.npz')
    p.add_argument('--render',action='store_true')
    a=p.parse_args();evaluate(a.actors,a.out,a.tasks,a.seeds,a.duration,a.safety_filter,
         a.risk_abort,a.wall_time_limit_s,a.ood_dataset,a.render,a.hard_timeout_s)
if __name__=='__main__':main()

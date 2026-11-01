"""Benchmarks teacher and deployed student under the *same live plant snapshots*.

The reference NMPC is the applied teacher; student actions are evaluated
counterfactually at the same observations, with the local progress and speed
filter active. Reports latencies ONLY, not student outcome on those states.
CPU, sequential decision, excludes plant integration and I/O. Results are
machine-dependent, not a proof of real-time deployment on embedded hardware.
"""
import os
for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[var]='1'
import json, math, time, argparse, platform, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from marl_game.env import EnvConfig,BerthingGameEnv
from model.ship import Ship
from ship_state_local.sensors import ShipStateLocalSensor
from decentralized_nmpc import LocalNMPC,NMPCConfig
from imitation.context_routed import ContextRoutedActors
from imitation.features import from_observation
from imitation.progress_assist import LocalProgressAssist
from imitation.safety_filter import LocalActionFilter

def measure(root,out,n=18,burn=3,seed=123):
    torch.set_num_threads(1)
    base=root/'il_results/stage3/actors_balanced'
    adverse=root/'il_results/stage4/actors_paper_candidate'
    if not base.exists() or not adverse.exists(): raise FileNotFoundError('Missing archived actor weights')
    learners=ContextRoutedActors(base,adverse)
    records=[]
    for wind_dir,current_dir in ((90,90),(270,270)):
        Ship.DEFAULT_CORIOLIS_MODE='energy_consistent'
        cfg=EnvConfig(train=False,initial_psi_deg=20.,max_seconds=(burn+n+5)*10.,dt=.2,decision_seconds=10.,seed=seed,target_x=50.,target_y=-100.,target_psi_deg=0.,eval_start_x_m=0.,eval_start_y_m=0.,eval_wind_mps=8.,eval_wind_direction_deg=wind_dir,eval_current_mps=.25,eval_current_direction_deg=current_dir,post_success_seconds=0.,stop_on_unsafe=True,hold_s=5.,push_heading_guard=True)
        env=BerthingGameEnv(cfg);env.reset()
        env.plant.env=(8.,math.radians(wind_dir),.25,math.radians(current_dir))
        sensors=[ShipStateLocalSensor(i,seed=seed) for i in range(4)]
        ncfg=NMPCConfig(horizon_s=90.,max_iterations=6,environmental_thrust_reserve=True,mission_guidance=True,disturbance_mode='measured',berth_position_tolerance_m=2.5)
        teachers=[LocalNMPC(i,env.goal,env.hull.points,env.plant.attach_local[i],env.side,env.plant.max_force,ncfg) for i in range(4)]
        assists=[LocalProgressAssist(i+1,env.goal,toward_only=True) for i in range(4)]
        guards=[LocalActionFilter(i+1,env.goal,env.hull.points,env.plant.attach_local[i],env.side) for i in range(4)]
        prev_own=[None]*4;prev_time=[None]*4;prev_action=[np.array([0.,0.]) for _ in range(4)]
        for k in range(burn+n):
            obs=[s.read(env) for s in sensors]
            t0=time.perf_counter()
            student_action=[]
            for i in range(4):
                ft=from_observation(obs[i],env.goal,prev_action[i],prev_own[i],prev_time[i])
                y=learners.act(i+1,ft,prev_action[i])
                y,_=assists[i].apply(obs[i],y)
                y,_=guards[i].apply(obs[i],y,prev_action[i])
                student_action.append(y)
            dt_student=(time.perf_counter()-t0)*1000.
            t1=time.perf_counter()
            teacher_actions=[]
            for i in range(4):
                u,report=teachers[i].act(obs[i])
                teacher_actions.append(u)
            dt_teacher=(time.perf_counter()-t1)*1000.
            if k>=burn: records.append(dict(wind_to_deg=wind_dir,current_to_deg=current_dir,t_s=env.time,IL_cycle_ms=round(dt_student,6),NMPC_cycle_ms=round(dt_teacher,6)))
            for i in range(4):
                prev_own[i]=obs[i].own[:3].copy();prev_time[i]=float(obs[i].time_s)
                prev_action[i]=np.array([(teacher_actions[i][0]+1.)/2.,teacher_actions[i][1]*20.],dtype=float)
            env.step(np.asarray(teacher_actions,dtype=np.float32))
            if env.time>=(burn+n)*10-1e-4: break
    import csv
    out.mkdir(parents=True,exist_ok=True)
    with (out/'decision_latency_samples.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    a=np.asarray([r['IL_cycle_ms'] for r in records]); b=np.asarray([r['NMPC_cycle_ms'] for r in records])
    report={'measurement':'Sequential 4-tug control-cycle latency on common online teacher-generated states', 'samples':len(records),'burn_in_cycles_per_environment':burn,'measured_cycles_per_environment':n,'timestamp_unix':time.time(), 'python':platform.python_version(),'platform':platform.platform(),'cpu':platform.processor(), 'torch':torch.__version__,'device':'CPU','threads':1,'conditions':[{'wind_to_deg':90,'current_to_deg':90},{'wind_to_deg':270,'current_to_deg':270}], 'student_pipeline':'4 actor feature/forward + local progress/approach filters; weights loaded once', 'teacher_pipeline':'4 individually optimized NMPC controllers; warm started, maxiter=6','included':'Policy calculation and local preprocessing, excludes environment integration/disk I/O, excludes weight loading', 'median_IL_cycle_ms':float(np.median(a)),'p90_IL_cycle_ms':float(np.percentile(a,90)),'median_NMPC_cycle_ms':float(np.median(b)),'p90_NMPC_cycle_ms':float(np.percentile(b,90)),'ratio_of_medians':float(np.median(b)/np.median(a)),'paired_median_speedup':float(np.median(b/a)), 'limitations':'Only early approach states (first roughly 210s); timings depend on hardware, Python, solver, library versions, warm starts. Not a full mission latency guarantee and not a closed-loop outcome comparison.'}
    (out/'decision_latency_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,default=Path('benchmark/results'));p.add_argument('--cycles',type=int,default=18)
    a=p.parse_args();measure(a.root,a.out,a.cycles)

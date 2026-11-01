"""Closed-loop student evaluation and DAgger data collection on the original plant.

No peer-tug observation enters an actor: each student has its own memory,
its own finite-difference sensor features, and its own neural policy.
The simulator alone combines the four actions. The expert is queried ONLY in
DAgger mode. Simulation safety monitor remains active and stops on unsafe.
"""
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
from marl_game.env import EnvConfig,BerthingGameEnv
from ship_state_local.sensors import ShipStateLocalSensor
from decentralized_nmpc import LocalNMPC,NMPCConfig
from physical_v4.termination import BerthingMonitor,BerthingCriteria
from model.ship import Ship
from .features import from_observation
from .policy import IndependentActors


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(rows)


def simulate(actors_dir,out,psi0=20.,wind_mps=0.,wind_to_deg=0.,current_mps=0.,
             current_to_deg=0.,target_x=50.,target_y=-100.,max_seconds=6500.,
             seed=123,dt=.2,decision_seconds=10.,position_tolerance=2.5,
             beta=0.,query_expert=False,teacher_maxiter=6,render=False,verbose=True,
             start_x=0.,start_y=0.,device='cpu',policy_type='neural',retrieval_dataset=None,retrieval_neighbors=3,retrieval_cases=2,retrieval_temperature=.8,safety_filter=False,risk_abort=False,ood_dataset=None,wall_time_limit_s=None,
             wind_sensor_std_mps=0.,current_sensor_std_mps=0.,ship_mass_scale=1.,
             progress_assist=False,progress_assist_toward_only=False,base_actors=None):
    if not 0<=beta<=1:raise ValueError('beta must be 0..1')
    if beta>0 and not query_expert:raise ValueError('beta>0 requires query_expert=True')
    if risk_abort and not safety_filter:raise ValueError('risk_abort requires safety_filter=True')
    if wall_time_limit_s is not None and wall_time_limit_s<=0:raise ValueError('wall_time_limit_s must be positive')
    if not (.6<=ship_mass_scale<=1.4):raise ValueError('ship_mass_scale must lie in [0.6,1.4]')
    if min(wind_sensor_std_mps,current_sensor_std_mps)<0:raise ValueError('sensor std must be nonnegative')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    Ship.DEFAULT_CORIOLIS_MODE='energy_consistent'
    cfg=EnvConfig(train=False,initial_psi_deg=psi0,max_seconds=max_seconds+10.,
                  dt=dt,decision_seconds=decision_seconds,seed=seed,
                  target_x=target_x,target_y=target_y,target_psi_deg=0.,
                  eval_start_x_m=start_x,eval_start_y_m=start_y,
                  eval_wind_mps=wind_mps,eval_wind_direction_deg=wind_to_deg,
                  eval_current_mps=current_mps,eval_current_direction_deg=current_to_deg,
                  post_success_seconds=0.,stop_on_unsafe=True,hold_s=5.,push_heading_guard=True)
    env=BerthingGameEnv(cfg);env.reset()
    # Deliberately apply *rigid-body* ship mass / inertia shift to the PLANT only.
    # Added mass and hydrodynamic coefficients remain nominal; this is one
    # controlled out-of-distribution test, not arbitrary unknown dynamics.
    env.ship_model.m0*=ship_mass_scale
    env.ship_model.IzG0*=ship_mass_scale
    env.monitor=BerthingMonitor(BerthingCriteria(position_m=position_tolerance,hold_s=5.))
    env.plant.env=(wind_mps,math.radians(wind_to_deg),current_mps,math.radians(current_to_deg))
    if policy_type=='neural':
        actors=IndependentActors(actors_dir,device=device)
    elif policy_type=='routed-neural':
        from .context_routed import ContextRoutedActors
        if base_actors is None:raise ValueError('routed-neural requires --base-actors')
        actors=ContextRoutedActors(base_actors,actors_dir,device=device)
    elif policy_type=='retrieval':
        from .retrieval import RetrievalActors
        actors=RetrievalActors(retrieval_dataset or 'il_results/expert_dataset.npz',neighbors=retrieval_neighbors)
    elif policy_type=='scenario-memory':
        from .case_library import ScenarioMemoryActors
        actors=ScenarioMemoryActors(retrieval_dataset or 'il_results/expert_dataset.npz',
                    neighbors=retrieval_neighbors,cases=retrieval_cases,context_temperature=retrieval_temperature)
    else:raise ValueError('Unknown policy type: '+policy_type)
    sensors=[ShipStateLocalSensor(i,seed=seed,wind_sensor_std_mps=wind_sensor_std_mps,
             current_sensor_std_mps=current_sensor_std_mps) for i in range(4)]
    assists=None
    if progress_assist:
        from .progress_assist import LocalProgressAssist
        assists=[LocalProgressAssist(i+1,env.goal,toward_only=progress_assist_toward_only) for i in range(4)]
    assist_log=[]
    guards=None
    if safety_filter:
        from .safety_filter import LocalActionFilter
        guards=[LocalActionFilter(i+1,env.goal,env.hull.points,env.plant.attach_local[i],env.side) for i in range(4)]
    expert=None
    if query_expert:
        teacher_cfg=NMPCConfig(horizon_s=90.,max_iterations=teacher_maxiter,
                               environmental_thrust_reserve=True,mission_guidance=True,
                               disturbance_mode='measured',berth_position_tolerance_m=position_tolerance)
        expert=[LocalNMPC(i,env.goal,env.hull.points,env.plant.attach_local[i],
                          env.side,env.plant.max_force,teacher_cfg) for i in range(4)]
    rng=np.random.default_rng(seed+7551)
    previous_own=[None]*4;previous_time=[None]*4;previous_action=[np.array([0.,0.]) for _ in range(4)]
    observations=[];actions_rows=[];trajectory=[];samples=[]; labels=[];sample_agents=[];sample_times=[]
    impulse=0.;peak_force=0.;min_gap=1e9;inf_ms=[];expert_ms=[]
    reason='TIME_GUARD';information={};initial_perf=time.perf_counter()
    safety_log=[]
    ood_results=[]
    support_monitor=None
    if ood_dataset:
        from .ood_monitor import EnvironmentalSupport
        support_monitor=[EnvironmentalSupport(ood_dataset,agent_id=i+1) for i in range(4)]

    def record(t,env_actions=None,step_info=None):
        sh=env.ship;envstate=env.plant.env
        err=float(np.linalg.norm(sh[:2]-env.goal[:2]));psi_deg=math.degrees(sh[2]);yaw=math.degrees(sh[5])
        c,s=math.cos(sh[2]),math.sin(sh[2]);vg=np.array([c*sh[3]-s*sh[4],s*sh[3]+c*sh[4]])
        points=env.plant.attachment(sh)
        d=dict(time_s=float(t),x_m=float(sh[0]),y_m=float(sh[1]),psi_deg=psi_deg,
               u_mps=float(sh[3]),v_mps=float(sh[4]),r_deg_s=yaw,
               position_error_m=err,heading_error_deg=abs((psi_deg+180)%360-180),
               hull_gap_m=float(env.hull_gap()),ship_speed_mps=float(np.linalg.norm(sh[3:5])),
               east_speed_mps=float(vg[0]),north_speed_mps=float(vg[1]),
               approach_speed_mps=float(max(0.,-vg[1])),
               wind_mps=float(envstate[0]),wind_to_deg=math.degrees(envstate[1]),
               current_mps=float(envstate[2]),current_to_deg=math.degrees(envstate[3]))
        for i in range(4):
            j=i+1;own=env.tugs[i];tugpsi=own[2]
            ap=env.plant.ap if i<2 else env.plant.at
            anchor=own[:2]+np.array([math.cos(tugpsi)*ap,math.sin(tugpsi)*ap])
            # Existing renderer expects physical rope anchor and ship contact separately.
            d.update({f'tug{j}_x_m':float(own[0]),f'tug{j}_y_m':float(own[1]),
                      f'tug{j}_psi_deg':math.degrees(own[2]),
                      f'tug{j}_contact_x_m':float(points[i][0]),
                      f'tug{j}_contact_y_m':float(points[i][1]),
                      f'tug{j}_rope_anchor_x_m':float(anchor[0]),
                      f'tug{j}_rope_anchor_y_m':float(anchor[1]),
                      f'tug{j}_thrust_N':float(env.act.thrust[i]),
                      f'tug{j}_fraction':float(previous_action[i][0]),
                      f'tug{j}_offset_deg':float(previous_action[i][1])})
        if step_info is not None:
            for k,idx in [('push1_N',0),('push2_N',1)]:d[k]=float(step_info['push_normal_N'][idx])
            for k,idx in [('tow3_N',0),('tow4_N',1)]:d[k]=float(step_info['tow_tension_N'][idx])
        return d

    trajectory.append(record(0.))
    while env.time<max_seconds-1e-8:
        if wall_time_limit_s is not None and time.perf_counter()-initial_perf>wall_time_limit_s:
            reason='WALL_TIME_ABORT'
            information=dict(unsafe=False,reason='COMPUTATIONAL_WATCHDOG',
                position_error_m=float(np.linalg.norm(env.ship[:2]-env.goal[:2])),
                heading_error_deg=float(abs(np.rad2deg(env.ship[2]))))
            break
        obs=[s.read(env) for s in sensors]
        chosen=[];report=[]
        # In DAgger, mix at the *time-step* level rather than mixing some
        # teacher tugs with student tugs within a single plant step. The
        # executed four-agent action is still a set of independent decisions.
        use_teacher_step=bool(query_expert and rng.random()<beta)
        for i in range(4):
            feat=from_observation(obs[i],env.goal,previous_action[i],previous_own[i],previous_time[i])
            if support_monitor is not None and len(ood_results)<4:
                ood_results.append(dict(agent=i+1,**support_monitor[i].evaluate(feat)))
            tic=time.perf_counter()
            stu=actors.act(i+1,feat,previous_action[i])
            inf_ms.append((time.perf_counter()-tic)*1000.)
            exp=None
            if query_expert:
                texp=time.perf_counter()
                teact,tdiag=expert[i].act(obs[i]);expert_ms.append((time.perf_counter()-texp)*1000.)
                exp=np.array([(float(teact[0])+1.)*.5,float(teact[1])*20.])
                samples.append(feat);labels.append(np.array([exp[0],exp[1]/20.],dtype=np.float32))
                sample_agents.append(i+1);sample_times.append(env.time)
            use_expert=use_teacher_step
            physical=exp.copy() if use_expert else stu.copy()
            if not np.isfinite(physical).all():raise FloatingPointError('Nonfinite local actor action')
            # Local policy clipping remains within the same actuator bounds as teacher.
            physical[0]=float(np.clip(physical[0],0.,.9 if i<2 else .65))
            physical[1]=float(np.clip(physical[1],-20.,20.))
            if assists is not None:
                physical,assist_diag=assists[i].apply(obs[i],physical)
                assist_log.append(dict(t_s=float(env.time),agent=i+1,**assist_diag))
            guard_diag=None
            if guards is not None:
                physical,guard_diag=guards[i].apply(obs[i],physical,previous_action[i])
                safety_log.append(dict(t_s=float(env.time),agent=i+1,**guard_diag))
            chosen.append(physical)
            report.append(dict(t_s=float(env.time),agent=i+1,source='expert' if use_expert else 'student',
                               student_fraction=float(stu[0]),student_offset_deg=float(stu[1]),
                               expert_fraction=(float(exp[0]) if exp is not None else ''),
                               expert_offset_deg=(float(exp[1]) if exp is not None else ''),
                               fraction=float(physical[0]),offset_deg=float(physical[1])))
            # Avoid expert warm-start/residual pretending its command was executed.
            if query_expert and not use_expert:
                expert[i].last=physical.copy()
                expert[i].plan[:]=physical
                expert[i].last_own_force=expert[i]._force_signature(*physical)
            # Own causal sensor/action history is updated to the actual applied choice.
            previous_own[i]=obs[i].own[:3].copy()
            previous_time[i]=float(obs[i].time_s)
            previous_action[i]=physical.copy()
        # Optional fail-closed numerical experiment: halt BEFORE a plant step
        # if already close to the simulator's known fast-quay-approach limit.
        # This avoids silently calling an aborted mission a safe successful berth.
        # It is not a physically validated emergency-stop actuator/controller.
        if risk_abort and safety_log:
            last=safety_log[-4:]
            serious_quay=any(z['quay_gap_m']<52. and z['closing_speed_mps']>.09 for z in last)
            serious_tow=any(z['agent']>=3 and z['tow_side_margin_m']<8. for z in last)
            if serious_quay or serious_tow:
                reason='RISK_ABORT'
                information=dict(unsafe=False,reason=('EARLY_QUAY_RISK' if serious_quay else 'TOW_SIDE_RISK'),
                                 position_error_m=float(np.linalg.norm(env.ship[:2]-env.goal[:2])),
                                 heading_error_deg=float(abs(np.rad2deg(env.ship[2]))))
                break
        env_commands=np.asarray([actors.to_env_action(v) for v in chosen],dtype=np.float32)
        _,_,reward,terminated,truncated,info=env.step(env_commands)
        information=info
        step_duration=float(env.time-trajectory[-1]['time_s'])
        impulse+=float(np.abs(env.act.thrust).sum())*step_duration
        peak_force=max(peak_force,float(info.get('max_interaction_N',0.)))
        min_gap=min(min_gap,float(env.hull_gap()))
        actions_rows.extend(report)
        trajectory.append(record(env.time,env_commands,info))
        if verbose and (len(trajectory)%40==0 or terminated or truncated):
            print(f'[IL] t={env.time:7.1f} s; error={info.get("position_error_m",float("nan")):.2f} m; '
                  f'heading={info.get("heading_error_deg",float("nan")):.2f} deg',flush=True)
        if terminated or truncated:
            reason='UNSAFE' if info.get('unsafe',False) else 'SUCCESS' if info.get('success',False) else 'TIMEOUT'
            break
    if env.time>=max_seconds-1e-8 and reason=='TIME_GUARD':reason='TIMEOUT'
    csv_write(out/'trajectory.csv',trajectory)
    csv_write(out/'local_il_decisions.csv',actions_rows)
    if safety_filter:csv_write(out/'local_safety_filter.csv',safety_log)
    if progress_assist:csv_write(out/'local_progress_assist.csv',assist_log)
    if query_expert and samples:
        np.savez_compressed(out/'dagger_samples.npz',X=np.asarray(samples,dtype=np.float32),
             Y=np.asarray(labels,dtype=np.float32),agent=np.asarray(sample_agents,dtype=np.int32),
             t_s=np.asarray(sample_times,dtype=np.float32))
    metrics=dict(controller_method='independent_il_dagger_mixture' if query_expert else 'independent_imitation_learning',
                 policy_type=policy_type,
                 success=bool(env.monitor.first_success_time_s is not None),
                 unsafe=bool(information.get('unsafe',False)),status=reason,seconds=float(env.time),
                 position_error_m=float(information.get('position_error_m',np.linalg.norm(env.ship[:2]-env.goal[:2]))),
                 heading_error_deg=float(information.get('heading_error_deg',abs(math.degrees(env.ship[2])))),
                 min_logged_hull_gap_m=float(min_gap),max_interaction_force_N=float(peak_force),
                 integrated_abs_thrust_Ns=float(impulse),
                 mean_actor_inference_ms=float(np.mean(inf_ms)) if inf_ms else None,
                 mean_teacher_query_ms=float(np.mean(expert_ms)) if expert_ms else None,
                 dagger_beta=float(beta),expert_labels_collected=len(samples),
                 safety_filter_enabled=bool(safety_filter),risk_abort_enabled=bool(risk_abort),
                 progress_assist_enabled=bool(progress_assist),
                 progress_assist_toward_only=bool(progress_assist_toward_only),
                 route_counts=getattr(actors,'counts',None),
                 progress_assist_interventions=sum(a.count for a in assists) if assists else 0,
                 computational_watchdog_limit_s=wall_time_limit_s,
                 safety_filter_interventions=int(sum(g.count for g in guards)) if guards else 0,
                 max_quay_approach_risk=max((g.max_risk for g in guards),default=0.) if guards else 0.,
                 min_tow_side_margin_m=min((g.min_tow_side_margin for g in guards[2:]),default=None) if guards else None,
                 unsafe_reason=information.get('reason',''),
                 environment_support=ood_results if support_monitor is not None else None,
                 research_caveat='RISK_ABORT is failed mission halted by a numerical warning, not evidence of physical stopping ability or safety guarantee',
                 initial_heading_deg=float(psi0),goal_xy_m=[float(target_x),float(target_y)],
                 wind_mps=float(wind_mps),wind_to_deg=float(wind_to_deg),
                 current_mps=float(current_mps),current_to_deg=float(current_to_deg),
                 seed=int(seed),cpu_elapsed_s=round(time.perf_counter()-initial_perf,2),
                 ship_mass_scale=float(ship_mass_scale),
                 wind_sensor_std_mps=float(wind_sensor_std_mps),
                 current_sensor_std_mps=float(current_sensor_std_mps),
                 note='Student closed-loop result on full five-vessel simulator. No real-world validation or sustained holding inferred.')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    if render:
        # Uses original verified renderer; same path and four-time panels as NMPC.
        from plot_marl import _read_rows,_cfg_from_args_and_rows,_convert_rows_for_renderer,plot_four_stages,plot_trajectory_with_footprints
        raw=_read_rows(out/'trajectory.csv')
        class A:pass
        arg=A();arg.target_x=target_x;arg.target_y=target_y;arg.target_psi_deg=0.
        rcfg=_cfg_from_args_and_rows(arg,raw)
        rr=_convert_rows_for_renderer(raw,rcfg)
        plot_four_stages(rr,rcfg,out)
        plot_trajectory_with_footprints(raw,out,target=(target_x,target_y),target_psi_deg=0.)
    if verbose:print(json.dumps(metrics,indent=2),flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actors',default='il_results/bc');p.add_argument('--out',default='il_results/eval')
    p.add_argument('--psi0',type=float,default=20.);p.add_argument('--target-x',type=float,default=50.)
    p.add_argument('--target-y',type=float,default=-100.)
    p.add_argument('--wind-mps',type=float,default=8.);p.add_argument('--wind-to-deg',type=float,default=90.)
    p.add_argument('--current-mps',type=float,default=.25);p.add_argument('--current-to-deg',type=float,default=90.)
    p.add_argument('--start-x',type=float,default=0.);p.add_argument('--start-y',type=float,default=0.)
    p.add_argument('--duration',type=float,default=6500.);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--render',action='store_true');p.add_argument('--quiet',action='store_true')
    p.add_argument('--safety-filter',action='store_true',help='Enable heuristic local approach/side-margin filter; not formally certified')
    p.add_argument('--risk-abort',action='store_true',help='Abort simulation proactively on warning; counts as mission failure, not successful safety control')
    p.add_argument('--progress-assist',action='store_true',help='Optional ablation: local terminal progress correction, NOT pure neural IL')
    p.add_argument('--progress-assist-toward-only',action='store_true',help='Gate local terminal trim on disturbance projection toward the berth')
    p.add_argument('--base-actors',default=None,help='Standard expert checkpoint for routed-neural ensemble')
    p.add_argument('--ood-dataset',default=None,help='Training-only dataset for environment support diagnostic, no action modifications')
    p.add_argument('--wall-time-limit-s',type=float,default=None,help='Computational watchdog; aborted mission is a failure')
    p.add_argument('--wind-sensor-std-mps',type=float,default=0.)
    p.add_argument('--current-sensor-std-mps',type=float,default=0.)
    p.add_argument('--ship-mass-scale',type=float,default=1.)
    p.add_argument('--policy-type',choices=['neural','routed-neural','retrieval','scenario-memory'],default='neural')
    p.add_argument('--retrieval-dataset',default='il_results/expert_dataset.npz')
    p.add_argument('--retrieval-neighbors',type=int,default=3)
    p.add_argument('--retrieval-cases',type=int,default=2)
    p.add_argument('--retrieval-temperature',type=float,default=.8)
    a=p.parse_args()
    simulate(a.actors,a.out,a.psi0,a.wind_mps,a.wind_to_deg,a.current_mps,a.current_to_deg,
         a.target_x,a.target_y,a.duration,a.seed,render=a.render,verbose=not a.quiet,
         start_x=a.start_x,start_y=a.start_y,device=a.device,policy_type=a.policy_type,
         retrieval_dataset=a.retrieval_dataset,retrieval_neighbors=a.retrieval_neighbors,
         retrieval_cases=a.retrieval_cases,retrieval_temperature=a.retrieval_temperature,
         safety_filter=a.safety_filter,risk_abort=a.risk_abort,ood_dataset=a.ood_dataset,
         wall_time_limit_s=a.wall_time_limit_s,wind_sensor_std_mps=a.wind_sensor_std_mps,
         current_sensor_std_mps=a.current_sensor_std_mps,ship_mass_scale=a.ship_mass_scale,
         progress_assist=a.progress_assist,progress_assist_toward_only=a.progress_assist_toward_only,
         base_actors=a.base_actors)
if __name__=='__main__':main()

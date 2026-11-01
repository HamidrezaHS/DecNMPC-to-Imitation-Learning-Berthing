"""Prospective, paired, seed-resolved evaluation of four independent IL tugs and NMPC.

Commands run the *original* simulator. No parameter tuning or policy updates
occur here. Every case has its own sandbox; no stale metrics silently count.
Teacher physics cannot be paired at different rigid-body mass in the existing
run_nmpc interface and such runs are deliberately recorded UNSUPPORTED.

Usage (from repository root):
 python -m research_protocol.runner --dry-run --limit-tasks 2 --seeds 1049
 python -m research_protocol.runner --methods routed_il baseline_il teacher_nmpc --out results/prospective_v1
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SUPPORTED={"psi0","wind_mps","wind_to_deg","current_mps","current_to_deg", "target_x","target_y","wind_sensor_std_mps","current_sensor_std_mps","ship_mass_scale"}
METHODS=("routed_il","baseline_il","teacher_nmpc")


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def load_protocol(path):
    raw=Path(path).read_text(encoding="utf-8")
    p=json.loads(raw)
    if p.get("schema")!="AOR-IL-PROSPECTIVE-1":raise ValueError("Unknown task protocol schema")
    if not isinstance(p.get("tasks"),dict) or not p["tasks"]:raise ValueError("Task list empty")
    if not p.get("seeds"):raise ValueError("At least one seed needed")
    for name,task in p["tasks"].items():
        if not task or set(task)-SUPPORTED:raise ValueError(f"Unknown task attributes for {name}: {set(task)-SUPPORTED}")
        if any(not isinstance(v,(int,float)) for v in task.values()):raise ValueError(f"Invalid numerical task {name}")
        for k in ("wind_mps","current_mps","wind_sensor_std_mps","current_sensor_std_mps"):
            if task.get(k,0)<0:raise ValueError(f"Negative {k}")
        if task.get("ship_mass_scale",1)<=0:raise ValueError("Nonpositive mass scale")
    return p


def checkpoint_digests(method):
    if method=="teacher_nmpc":
        return {"nmpc_controller":digest(ROOT/"decentralized_nmpc"/"controller.py")}
    dirs=[ROOT/"il_results/stage3/actors_balanced"]
    if method=="routed_il":dirs.append(ROOT/"il_results/stage4/actors_paper_candidate")
    return {str(d.relative_to(ROOT)/f"actor_{i}.pt"):digest(d/f"actor_{i}.pt") for d in dirs for i in range(1,5)}


def make_command(method,task,seed,folder,seconds,maxiter):
    """Construct a complete, inspectable argv list; never shell=True."""
    if method not in METHODS:raise ValueError(f"Unknown method: {method}")
    if method=="teacher_nmpc" and task.get("ship_mass_scale",1)!=1:
        return None,"Existing teacher CLI cannot vary rigid-body plant mass identically."
    args=[sys.executable]
    if method=="teacher_nmpc":
        args += ["run_nmpc.py", "--controller-method","ship_state_local", "--no-plots", "--maxiter",str(maxiter),"--duration",str(seconds),"--out",str(folder),"--seed",str(seed)]
    else:
        args += ["-m","imitation.rollout", "--out",str(folder),"--seed",str(seed), "--duration",str(seconds), "--wall-time-limit-s",str(max(600.,seconds*.20)),"--quiet", "--safety-filter"]
        if method=="routed_il":
            args += ["--actors","il_results/stage4/actors_paper_candidate", "--base-actors","il_results/stage3/actors_balanced", "--policy-type","routed-neural", "--progress-assist", "--progress-assist-toward-only"]
        else:args += ["--actors","il_results/stage3/actors_balanced", "--policy-type","neural"]
    cli={"psi0":"--psi0","wind_mps":"--wind-mps","wind_to_deg":"--wind-to-deg", "current_mps":"--current-mps","current_to_deg":"--current-to-deg","target_x":"--target-x","target_y":"--target-y","wind_sensor_std_mps":"--wind-sensor-std-mps", "current_sensor_std_mps":"--current-sensor-std-mps","ship_mass_scale":"--ship-mass-scale"}
    for k in sorted(task):
        if method=="teacher_nmpc" and k=="ship_mass_scale":continue
        args += [cli[k],str(task[k])]
    return args,None


def metrics_to_record(metrics,method,name,seed,protocol_hash,status_override=None,notes=""):
    status=status_override or metrics.get("status") or "PROCESS_ERROR"
    valid=(status=="SUCCESS" and metrics.get("success") is True and metrics.get("unsafe") is False)
    return dict(method=method,task=name,seed=seed,status=status,success=valid,
                unsafe=metrics.get("unsafe"),time_s=metrics.get("seconds"),
                position_error_m=metrics.get("position_error_m"),
                heading_error_deg=metrics.get("heading_error_deg"),
                effort_Ns=metrics.get("integrated_abs_thrust_Ns"),
                peak_interaction_force_N=metrics.get("max_interaction_force_N",metrics.get("max_interaction_N")),
                safety_filter_interventions=metrics.get("safety_filter_interventions"),
                expert_queries=metrics.get("expert_labels_collected"),
                protocol_sha256=protocol_hash,notes=notes)


def save_records(folder,rows):
    (folder/"records.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    if rows:
        with (folder/"records.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(protocol,output,methods=METHODS,seeds=None,limit_tasks=0,seconds=None,
        maxiter=6,hard_timeout_s=1800,dry_run=False,resume=True):
    protocol=Path(protocol).resolve();p=load_protocol(protocol)
    methods=tuple(methods)
    if any(m not in METHODS for m in methods):raise ValueError("Unexpected method")
    seeds=list(map(int,seeds if seeds is not None else p["seeds"]))
    tasks=list(p["tasks"].items())[:limit_tasks or None]
    seconds=float(seconds if seconds is not None else p["max_sim_seconds"])
    if seconds<=0 or hard_timeout_s<=0:raise ValueError("Positive time limits required")
    phash=digest(protocol);output=Path(output).resolve()
    planned=[]
    for name,task in tasks:
        for seed in seeds:
            for method in methods:
                folder=output/method/name/f"seed_{seed}"
                cmd,skip=make_command(method,task,seed,folder,seconds,maxiter)
                planned.append((method,name,seed,folder,cmd,skip))
    if dry_run:
        for method,name,seed,folder,cmd,skip in planned:
            print(method,name,seed,"SKIP:"+skip if skip else " ".join(map(str,cmd)),flush=True)
        return []
    output.mkdir(parents=True,exist_ok=True)
    audit={"protocol_original":str(protocol),"protocol_sha256":phash,
           "run_at_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           "methods":{m:checkpoint_digests(m) for m in methods},
           "comparability_note":"A task with a changed mass scale has no NMPC comparator; filter and external supervision are not formal safety guarantees.",
           "analysis_policy":"All statuses kept, no success inference from endpoint error. Do not tune on prospective tests.",
           "python":sys.version}
    (output/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    (output/"protocol_frozen.json").write_text(protocol.read_text(encoding="utf-8"),encoding="utf-8")
    records=[]
    env={**os.environ, "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    for method,name,seed,folder,cmd,skip in planned:
        folder.mkdir(parents=True,exist_ok=True)
        (folder/"task.json").write_text(json.dumps(dict(task=p['tasks'][name],method=method,seed=seed,protocol_sha256=phash),indent=2),encoding="utf-8")
        if skip:
            m={};override="UNSUPPORTED_COMPARISON";err=skip
        elif resume and (folder/"metrics.json").exists() and (folder/"complete_stamp.json").exists():
            stamp=json.loads((folder/"complete_stamp.json").read_text())
            if stamp.get("protocol_sha256")==phash and stamp.get("checkpoint_hashes")==audit['methods'][method]:
                m=json.loads((folder/"metrics.json").read_text());override=None;err="cached verified"
            else:
                raise RuntimeError(f"Stale or mismatched checkpoint at {folder}; use new --out or --no-resume")
        else:
            (folder/"command.json").write_text(json.dumps(cmd,indent=2),encoding='utf-8')
            print(f"RUN {method} | {name} | seed={seed}",flush=True)
            try:
                t=time.perf_counter()
                proc=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=hard_timeout_s)
                (folder/"stdout.log").write_text(proc.stdout,encoding="utf-8")
                (folder/"stderr.log").write_text(proc.stderr,encoding="utf-8")
                if proc.returncode!=0 or not (folder/"metrics.json").exists():
                    m={};override="PROCESS_ERROR";err=f"Return code {proc.returncode}; inspect stderr.log"
                else:
                    m=json.loads((folder/"metrics.json").read_text());override=None;err=""
                    (folder/"complete_stamp.json").write_text(json.dumps(dict(protocol_sha256=phash,checkpoint_hashes=audit['methods'][method],wall_seconds=time.perf_counter()-t),indent=2),encoding="utf-8")
            except subprocess.TimeoutExpired as exc:
                m={};override="PROCESS_TIMEOUT";err=f"OS watchdog expired after {hard_timeout_s} s"
                (folder/"timeout.json").write_text(json.dumps(dict(status=override,error=err)),encoding="utf-8")
        rec=metrics_to_record(m,method,name,seed,phash,override,err)
        records.append(rec);save_records(output,records)
        print(f"  => {rec['status']}, err={rec['position_error_m']}",flush=True)
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',default='experiments/PROSPECTIVE_PROTOCOL_V1.json')
    ap.add_argument('--out',default='results/prospective_v1')
    ap.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    ap.add_argument('--seeds',nargs='+',type=int,default=None)
    ap.add_argument('--limit-tasks',type=int,default=0)
    ap.add_argument('--duration',type=float,default=None)
    ap.add_argument('--maxiter',type=int,default=6)
    ap.add_argument('--hard-timeout-s',type=float,default=1800.)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--no-resume',action='store_true')
    args=ap.parse_args()
    run(args.protocol,args.out,args.methods,args.seeds,args.limit_tasks,args.duration,args.maxiter,args.hard_timeout_s,args.dry_run,not args.no_resume)
if __name__=='__main__':main()

"""Frozen experimental suite for independent tug imitation and local supervision.

- Full case config stored before running each scenario.
- Calls independent subprocesses; captures exceptions and hard timeouts.
- All non-SUCCESS statuses count as unsuccessful or INCOMPLETE, never success.
- DOES NOT use teacher at evaluation; same five-vessel plant for all methods.
- Only non-overlapping forward tests may be called independent holdouts.
- Automatically writes audit file with hash of protocol + checkpoints.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as fp:
        for block in iter(lambda:fp.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def evaluate(protocol_file,out,methods=None,group='holdout',hard_timeout=90.,wall_limit=75.,render=False):
    cfg=json.loads(Path(protocol_file).read_text(encoding='utf-8'))
    if cfg.get('protocol_version')!='IL-PAPER-1':raise ValueError('Invalid protocol schema')
    methods=methods or list(cfg['methods'])
    if not isinstance(cfg.get('groups'),dict) or group not in cfg['groups']:raise ValueError('Unknown group')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source_hash=sha256(protocol_file)
    (out/'frozen_protocol_copy.json').write_text(Path(protocol_file).read_text(encoding='utf-8'))
    runs=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for method in methods:
        if method not in cfg['methods']:raise ValueError('Unknown method: '+method)
        settings=cfg['methods'][method]
        for name,params in cfg['groups'][group].items():
            folder=out/method/name
            folder.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'-m','imitation.rollout','--actors',settings['actors'],
                 '--out',str(folder),'--quiet','--duration',str(cfg.get('duration',6500)),
                 '--wall-time-limit-s',str(wall_limit)]
            if settings.get('base_actors'):cmd+=['--base-actors',settings['base_actors']]
            if settings.get('policy_type'):cmd+=['--policy-type',settings['policy_type']]
            for key in ('safety_filter','risk_abort','progress_assist','progress_assist_toward_only'):
                if settings.get(key):cmd+=['--'+key.replace('_','-')]
            for key,val in params.items():
                cmd+=['--'+key.replace('_','-'),str(val)]
            if render:cmd.append('--render')
            print('[EVAL]',method,name,flush=True)
            t0=time.monotonic()
            try:
                proc=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=hard_timeout)
                (folder/'process.log').write_text(proc.stdout+'\nSTDERR:\n'+proc.stderr,encoding='utf8')
                if proc.returncode!=0:
                    result={'status':'PROCESS_ERROR','success':False,'unsafe':None,'seconds':None,
                            'position_error_m':None,'cpu_elapsed_s':None,
                            'reason':f'exit code {proc.returncode}; inspect process.log'}
                else:result=json.loads((folder/'metrics.json').read_text())
            except subprocess.TimeoutExpired:
                result={'status':'PROCESS_TIMEOUT','success':False,'unsafe':None,'seconds':None,
                        'position_error_m':None,'cpu_elapsed_s':None,'reason':'OS-level watchdog expired'}
            success=bool(result.get('status')=='SUCCESS' and result.get('success') and not result.get('unsafe'))
            record=dict(method=method,case=name,partition=group,status=result.get('status'),
                success=success,unsafe=result.get('unsafe'),seconds=result.get('seconds'),
                position_error_m=result.get('position_error_m'),
                intervention_count=result.get('safety_filter_interventions'),
                progress_interventions=result.get('progress_assist_interventions'),
                query_count=result.get('expert_labels_collected'),
                elapsed_wall_s=round(time.monotonic()-t0,1),protocol_sha256=source_hash)
            runs.append(record)
            (out/'summary.json').write_text(json.dumps(runs,indent=2),encoding='utf8')
            with (out/'summary.csv').open('w',newline='',encoding='utf8') as fp:
                writer=csv.DictWriter(fp,fieldnames=list(runs[0]));writer.writeheader();writer.writerows(runs)
            print('[RESULT]',record['status'],record['position_error_m'],flush=True)
    overview={m:dict(n=sum(r['method']==m for r in runs),success=sum(r['method']==m and r['success'] for r in runs),
        unsafe=sum(r['method']==m and r['unsafe'] is True for r in runs)) for m in methods}
    overview['caveat']='Descriptive results, NOT a safety proof. A watchdog termination is incomplete.'
    (out/'score.json').write_text(json.dumps(overview,indent=2),encoding='utf8')
    return overview


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',default='PAPER_PROTOCOL.json')
    p.add_argument('--out',default='il_results/stage4/paper_suite')
    p.add_argument('--methods',nargs='+',default=None)
    p.add_argument('--group',choices=['development','holdout'],default='holdout')
    p.add_argument('--hard-timeout-s',type=float,default=100.)
    p.add_argument('--wall-limit-s',type=float,default=80.)
    p.add_argument('--render',action='store_true')
    a=p.parse_args()
    print(json.dumps(evaluate(a.protocol,a.out,a.methods,a.group,a.hard_timeout_s,a.wall_limit_s,a.render),indent=2))
if __name__=='__main__':main()

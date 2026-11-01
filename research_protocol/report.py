"""All-outcome analysis without masking unsuccessful, unsafe or incomplete runs.

Only report a win/loss against NMPC when *both* controllers were run on the
same task and seed; no invented imputations for unsupported mass-shift tests.
"""
from __future__ import annotations
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

METHODS=('routed_il','baseline_il','teacher_nmpc')
SUCCESS='SUCCESS'

def read_records(path):
    p=Path(path)
    if p.is_dir():p=p/'records.json'
    return json.loads(p.read_text(encoding='utf-8'))


def safe_num(v):
    try:return float(v) if v is not None and math.isfinite(float(v)) else None
    except (ValueError,TypeError):return None


def summarize(rows):
    methods=sorted(set(r['method'] for r in rows))
    result={}
    for m in methods:
        rr=[r for r in rows if r['method']==m]
        complete=[r for r in rr if r['status'] not in ('UNSUPPORTED_COMPARISON','PROCESS_ERROR','PROCESS_TIMEOUT')]
        success=[r for r in complete if r['status']==SUCCESS and r['success'] is True]
        result[m]=dict(attempts=len(rr),completed=len(complete),successes=len(success),
                unsafe=sum(r['status']=='UNSAFE' for r in complete),
                timed_out=sum(r['status']=='TIMEOUT' for r in complete),
                incomplete=sum(r['status'] in ('PROCESS_ERROR','PROCESS_TIMEOUT','UNSUPPORTED_COMPARISON') for r in rr),
                median_success_time_s=float(np.median([x['time_s'] for x in success if safe_num(x['time_s']) is not None])) if success else None,
                caution='Do not infer statistical safety or generalization from a few task types.')
    bykey=defaultdict(dict)
    for r in rows:
        key=(r['task'],str(r['seed']));bykey[key][r['method']]=r
    paired=[]
    for (case,seed),items in sorted(bykey.items()):
        if 'teacher_nmpc' in items and 'routed_il' in items:
            a,b=items['routed_il'],items['teacher_nmpc']
            paired.append(dict(task=case,seed=seed,
                 il_status=a['status'],nmpc_status=b['status'],
                 il_success=a['success'],nmpc_success=b['success'],
                 il_time_s=a['time_s'],nmpc_time_s=b['time_s'],
                 comparable=(a['status'] not in ('PROCESS_TIMEOUT','PROCESS_ERROR') and b['status'] not in ('UNSUPPORTED_COMPARISON','PROCESS_TIMEOUT','PROCESS_ERROR'))))
    return dict(per_method=result,paired_teacher=paired,
                caveats=['Do not include unexecuted scenarios as new empirical results.',
                         'Mass-shift tests lack a matched teacher because the teacher CLI does not expose the same plant change.',
                         'First arrival within the terminal tolerance does not imply sustained berthing, fender contact or safety certification.',
                         'Multiple seeds in a small group of tasks do not establish generalization across all directions.'])


def figure(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    methods=[m for m in METHODS if any(r['method']==m for r in rows)]
    if not methods:return
    status_names=['SUCCESS','TIMEOUT','UNSAFE','RISK_ABORT','PROCESS_TIMEOUT','PROCESS_ERROR','UNSUPPORTED_COMPARISON','OTHER']
    colors=['#16856B','#D3A74A','#BA5050','#8461A1','#777988','#AC6485','#ADB1B5','#3C5369']
    fig,ax=plt.subplots(figsize=(8,4.6))
    bottom=np.zeros(len(methods),float)
    for label,color in zip(status_names,colors):
        values=[]
        for method in methods:
            subset=[r for r in rows if r['method']==method]
            values.append(sum((r['status']==label if label!='OTHER' else r['status'] not in status_names[:-1]) for r in subset))
        ax.bar(methods,values,bottom=bottom,color=color,label=label,width=.58)
        bottom+=np.asarray(values)
    ax.set_ylabel('Completed and incomplete trial counts')
    ax.set_title('Prospective paired tests: all outcomes are retained')
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left',bbox_to_anchor=(1.0,1.0),fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(out/'all_outcomes.pdf');fig.savefig(out/'all_outcomes.png',dpi=210);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='results/prospective_v1')
    args=p.parse_args();path=Path(args.results)
    rows=read_records(path)
    report=summarize(rows)
    (path/'analysis_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    figure(rows,path)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

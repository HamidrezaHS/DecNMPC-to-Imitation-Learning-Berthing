"""Publication-ready figure comparing matched recorded NMPC and student rollouts."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with Path(path).open(newline='',encoding='utf-8') as fp:
        r=list(csv.DictReader(fp))
    if not r:raise ValueError(f'No trajectory in {path}')
    return r

def get(rows,k):return np.array([float(r[k]) for r in rows])

def run(teacher_dir,student_dirs,out):
    cases=[('Dec-NMPC expert',Path(teacher_dir),'#1a5986')]
    for name,folder in student_dirs:
        cases.append((name,Path(folder), '#dd8a2a' if len(cases)==1 else '#8e56b9'))
    summary=[]
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.6),layout='constrained')
    for name,path,col in cases:
        rr=read(path/'trajectory.csv')
        m=json.loads((path/'metrics.json').read_text())
        t=get(rr,'time_s')/60.
        axes[0,0].plot(get(rr,'x_m'),get(rr,'y_m'),label=name,color=col,lw=1.65)
        axes[0,1].plot(t,get(rr,'position_error_m'),label=name,color=col,lw=1.65)
        axes[1,0].plot(t,get(rr,'heading_error_deg'),label=name,color=col,lw=1.65)
        axes[1,1].plot(t,get(rr,'hull_gap_m'),label=name,color=col,lw=1.65)
        summary.append(dict(method=name,status=m['status'],success=m['success'],unsafe=m['unsafe'],
                            time_s=m['seconds'],position_error_m=m['position_error_m']))
    axes[0,0].scatter([50],[-100],color='#e11d48',s=100,marker='*',zorder=6,label='Berth reference')
    axes[0,0].set(xlabel='East x (m)',ylabel='North y (m)',title='(a) Ship-centre path')
    axes[0,0].axis('equal')
    axes[0,1].axhline(2.5,lw=1,ls='--',color='#64748b',label='Position tolerance')
    axes[0,1].set(xlabel='Simulated time (min)',ylabel='Position error (m)',title='(b) Position convergence')
    axes[1,0].axhline(2.,lw=1,ls='--',color='#64748b',label='Heading tolerance')
    axes[1,0].set(xlabel='Simulated time (min)',ylabel='Heading error (deg)',title='(c) Heading regulation')
    axes[1,1].set(xlabel='Simulated time (min)',ylabel='Hull–quay gap (m)',title='(d) Quay clearance')
    axes[0,0].legend(fontsize=8,frameon=False)
    for ax in axes.flat:ax.grid(alpha=.18)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'matched_expert_vs_student.pdf',bbox_inches='tight')
    fig.savefig(out/'matched_expert_vs_student.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    (out/'matched_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',default='expert_archive/environment_4x4_v31/combined_psi20_w8_wd90_c0p25_cd90')
    p.add_argument('--student',default='il_results/eval_bc_90_90')
    p.add_argument('--dagger',default=None)
    p.add_argument('--out',default='il_results/compare')
    a=p.parse_args()
    ss=[('Behavior Cloning',a.student)]
    if a.dagger:ss.append(('DAgger (preliminary)',a.dagger))
    print(json.dumps(run(a.teacher,ss,a.out),indent=2))
if __name__=='__main__':main()

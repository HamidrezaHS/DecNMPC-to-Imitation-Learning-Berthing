"""Produce traceable manuscript-style plots comparing archived teacher and students.

Only reads recorded trajectories; does not rerun simulation or invent results.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE={'teacher':'#1b637f','original_bc':'#b65b62','terminal_bc':'#078b79','dagger':'#7a56a3'}

def load(folder):
    p=Path(folder)
    with (p/'trajectory.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError(f'No trajectory samples in {folder}')
    keys=rows[0]
    def col(key):return np.array([float(row[key]) for row in rows])
    return col,rows


def plot(teacher,original_bc,terminal_bc,out,dagger=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cases=[('Dec-NMPC expert',teacher,'teacher'),('Original BC',original_bc,'original_bc'),
           ('Terminal-weighted BC',terminal_bc,'terminal_bc')]
    if dagger is not None:cases.append(('DAgger (expert-free evaluation)',dagger,'dagger'))
    fig,ax=plt.subplots(2,2,figsize=(11.2,7.5),layout='constrained')
    for title,p,k in cases:
        col,raw=load(p);t=col('time_s')/60.
        color=PALETTE[k]
        ax[0,0].plot(col('x_m'),col('y_m'),lw=1.8,color=color,label=title)
        ax[0,1].plot(t,col('position_error_m'),lw=1.6,color=color,label=title)
        ax[1,0].plot(t,col('heading_error_deg'),lw=1.5,color=color,label=title)
        ax[1,1].plot(t,col('ship_speed_mps'),lw=1.5,color=color,label=title)
        ax[0,0].scatter([col('x_m')[-1]],[col('y_m')[-1]],s=40,facecolor=color,edgecolor='white',zorder=5)
    ax[0,0].scatter([50],[-100],marker='*',color='#30343b',s=160,label='Target berth',zorder=6)
    ax[0,0].set(xlabel='East x (m)',ylabel='North y (m)',title='(a) Ship-centre trajectory');ax[0,0].axis('equal')
    ax[0,1].axhline(2.5,ls='--',lw=1.2,color='#c78c32',label='2.5 m acceptance limit')
    ax[0,1].set(xlabel='Time (min)',ylabel='Position error (m)',title='(b) Position convergence')
    ax[1,0].axhline(2.,ls='--',lw=1.2,color='#c78c32')
    ax[1,0].set(xlabel='Time (min)',ylabel='Absolute heading error (deg)',title='(c) Heading convergence')
    ax[1,1].axhline(.045,ls='--',lw=1.2,color='#c78c32')
    ax[1,1].set(xlabel='Time (min)',ylabel='Ship planar speed (m/s)',title='(d) Speed evolution')
    for a in ax.flat:a.grid(alpha=.20)
    ax[0,0].legend(loc='best',frameon=False,fontsize=8)
    fig.suptitle('Matched 90°/90° environment: expert and independent imitation policies',fontsize=13,color='#263d4f',weight='bold')
    fig.savefig(out/'matched_teacher_bc_comparison.png',dpi=210,bbox_inches='tight')
    fig.savefig(out/'matched_teacher_bc_comparison.pdf',bbox_inches='tight')
    plt.close(fig)

    p=Path(terminal_bc)
    col,raw=load(p);t=col('time_s')/60.
    fig,ax=plt.subplots(2,1,figsize=(10.6,6.3),sharex=True,layout='constrained')
    cs=['#2780ac','#20a597','#b9792c','#7b6cc2']
    for i in range(1,5):
        ax[0].plot(t,col(f'tug{i}_fraction'),color=cs[i-1],lw=1.2,label=f'Tug {i}: {"push" if i<3 else "tow"}')
        ax[1].plot(t,col(f'tug{i}_thrust_N')/1e6,color=cs[i-1],lw=1.2,label=f'Tug {i}')
    ax[0].set_ylabel('Commanded thrust fraction')
    ax[1].set(xlabel='Simulation time (min)',ylabel='Applied tug thrust (MN)')
    for a in ax:a.grid(alpha=.23);a.legend(ncol=4,frameon=False,loc='upper right',fontsize=8)
    fig.suptitle('Refined BC: independently commanded tugs (no expert queries)',fontsize=13,weight='bold')
    fig.savefig(out/'refined_bc_tug_actions.png',dpi=210,bbox_inches='tight')
    fig.savefig(out/'refined_bc_tug_actions.pdf',bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',default='expert_archive/environment_4x4_v31/combined_psi20_w8_wd90_c0p25_cd90')
    p.add_argument('--original-bc',default='il_results/eval_bc_90_90')
    p.add_argument('--terminal-bc',default='il_results/stage2/neural_terminal_90')
    p.add_argument('--out',default='il_results/stage2/paper_figures')
    p.add_argument('--dagger',default='il_results/stage2/dagger_round2_combo')
    a=p.parse_args();print(plot(a.teacher,a.original_bc,a.terminal_bc,a.out,dagger=a.dagger))
if __name__=='__main__':main()

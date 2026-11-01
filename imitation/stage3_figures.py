"""Generate scientific figures from real stage-2/3 logs (no synthetic samples)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create(root='il_results/stage3',previous='il_results/stage2'):
    root=Path(root);prev=Path(previous);out=root/'figures';out.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42})
    def load(p):
        p=Path(p)
        if not p.is_file():return None
        return pd.read_csv(p)
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=220,bbox_inches='tight',pad_inches=.08)
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight',pad_inches=.08)
        plt.close(fig)
    base=load(prev/'dagger_round2_combo'/'trajectory.csv')
    new=load(root/'eval_balanced'/'main90'/'trajectory.csv')
    if base is not None and new is not None:
        fig,ax=plt.subplots(1,2,figsize=(10.6,3.9),layout='constrained')
        for d,name in [(base,'Stage-2 DAgger'),(new,'Stage-3 scenario-balanced')]:
            ax[0].plot(d.time_s/60.,d.position_error_m,label=name,lw=1.4)
            ax[1].plot(d.time_s/60.,d.heading_error_deg,label=name,lw=1.4)
        ax[0].axhline(2.5,ls='--',color='#808c9a',lw=1,label='Position acceptance threshold')
        ax[1].axhline(2.,ls='--',color='#808c9a',lw=1,label='Heading acceptance threshold')
        ax[0].set(xlabel='Simulation time (min)',ylabel='Ship position error (m)',title='(a) Position convergence')
        ax[1].set(xlabel='Simulation time (min)',ylabel='Heading error (deg)',title='(b) Heading convergence')
        for a in ax:a.grid(alpha=.20);a.legend(fontsize=7,frameon=False)
        save(fig,'Fig01_Matched_Stage2_vs_Stage3')
    scores=[('Wind/current 90°/90°',root/'eval_balanced'/'main90'/'metrics.json'),
            ('Initial heading 25°',root/'eval_balanced_fast'/'heading25'/'metrics.json'),
            ('Modified wind/current speed',root/'eval_balanced_fast'/'magnitude_new'/'metrics.json'),
            ('Mass +5%, sensor noise',root/'eval_mass105_noise'/'metrics.json')]
    vals=[];labels=[]
    for name,p in scores:
        if p.is_file():
            m=json.loads(p.read_text());
            if m.get('status')=='SUCCESS':labels.append(name);vals.append(m['seconds']/60.)
    if vals:
        fig,ax=plt.subplots(figsize=(8.3,3.9));y=np.arange(len(vals))
        ax.barh(y,vals,color='#226c84');ax.set_yticks(y,labels);ax.invert_yaxis()
        for i,v in enumerate(vals):ax.text(v+1,i,f'{v:.1f} min',va='center',fontsize=9)
        ax.set_xlim(0,max(vals)+17);ax.set_xlabel('Time to first sustained terminal acceptance (min)')
        ax.set_title('Successful expert-free neural actor evaluations (one seed)')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        save(fig,'Fig02_Successful_Conditions')
    comparisons=[('No filter',root/'eval_balanced_fast'/'heldout270'/'trajectory.csv',root/'eval_balanced_fast'/'heldout270'/'metrics.json'),
        ('Local command filter',root/'guard_w270_c270'/'trajectory.csv',root/'guard_w270_c270'/'metrics.json'),
        ('Filter + risk abort',root/'guard_abort_270'/'trajectory.csv',root/'guard_abort_270'/'metrics.json')]
    fig,ax=plt.subplots(figsize=(8.6,4.1))
    for name,tra,met in comparisons:
        d=load(tra)
        if d is None or not met.is_file():continue
        m=json.loads(met.read_text())
        ax.plot(d.time_s,d.position_error_m,label=f'{name}: {m["status"]} at {m["seconds"]:.0f} s',lw=1.4)
        ax.scatter([d.time_s.iloc[-1]],[d.position_error_m.iloc[-1]],s=24,zorder=5)
    ax.set(xlabel='Simulation time (s)',ylabel='Ship position error (m)',title='Direction 270°/270°: interventions delay, but do not resolve, failure')
    ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    save(fig,'Fig03_Adverse_270_Diagnostics')
    d=load(root/'actors_balanced'/'training_history.csv')
    if d is not None:
        fig,axs=plt.subplots(2,2,figsize=(9.4,6.3),layout='constrained')
        for i,ax in enumerate(axs.flat,1):
            sub=d[d.agent==i]
            ax.plot(sub.epoch,sub.train_loss,label='Training')
            ax.plot(sub.epoch,sub.val_loss,label='Validation')
            ax.set(title=f'Tug {i} ({"push" if i<3 else "tow"})',xlabel='Epoch',ylabel='Imitation loss')
            ax.grid(alpha=.2)
            if i==1:ax.legend(frameon=False)
        save(fig,'Fig04_Multidomain_Training')
    print('Completed figures in',out)

if __name__=='__main__':
    create()

"""Vector plots for IL publication draft from recorded simulation CSVs only."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

navy='#18334f';blue='#236e9a';green='#15827d';gold='#d58c36';red='#b14c5e';gray='#697d8d'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.2,'axes.labelsize':9.8,
    'axes.titlesize':11,'legend.fontsize':8.5,'pdf.fonttype':42,'savefig.dpi':280,
    'axes.spines.top':False,'axes.spines.right':False})
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'manuscript_il'/'Figures';OUT.mkdir(parents=True,exist_ok=True)


def save(fig,stem):
    fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight',pad_inches=.13)
    fig.savefig(OUT/f'{stem}.png',bbox_inches='tight',pad_inches=.13)
    plt.close(fig)

def R(path):
    return pd.read_csv(ROOT/path/'trajectory.csv')

# Figure 1: minimal local architecture
fig,ax=plt.subplots(figsize=(11.6,3.65));ax.set_xlim(0,12.4);ax.set_ylim(0,4.0);ax.axis('off')
def block(x,y,w,h,heading,detail,color,fill):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10,rounding_size=.14',
       facecolor=fill,edgecolor=color,linewidth=1.6))
    ax.text(x+w/2,y+h*.68,heading,ha='center',va='center',weight='bold',fontsize=10.3,color=navy)
    ax.text(x+w/2,y+h*.27,detail,ha='center',va='center',fontsize=8.1,color=navy,linespacing=1.2)
def arrow(x,y,xx,yy,color=gray,style='-|>',linestyle='-'):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle=style,mutation_scale=13,lw=1.55,
                                  linestyle=linestyle,color=color,connectionstyle='arc3'))
block(.15,1.48,2.50,1.25,'Local information','Ship-state feedback\nOwn tug + local wind/current',blue,'#eaf3f8')
block(3.03,2.34,2.4,.75,'Standard actor','One network per tug',blue,'#eaf3f8')
block(3.03,1.16,2.4,.75,'Adverse actor','One network per tug',gold,'#fff3e3')
block(5.92,1.50,2.36,1.22,'Environmental gate','Wind/current projected\ntoward berth',green,'#e8f5f2')
block(8.76,1.50,3.15,1.22,'Local command processing','Progress trim\n+ approach filter',green,'#e8f5f2')
arrow(2.7,2.18,3,2.64);arrow(2.7,1.8,3,1.54)
arrow(5.48,2.64,5.87,2.39);arrow(5.48,1.54,5.87,1.86)
arrow(8.33,2.11,8.70,2.11)
ax.text(6.1,3.58,'Same design runs independently on each of the four tugs',fontsize=11,weight='bold',color=navy,ha='center')
ax.text(6.1,.63,'The coupled five-vessel plant receives the four commands; only the ship state is shared.',fontsize=9,color=gray,ha='center')
ax.text(6.1,.25,'No peer-tug state or command exchange; no NMPC solver at inference',fontsize=9,color=gray,ha='center')
save(fig,'Fig01_IndependentLearningArchitecture')

# Figure 2: 90 vs 270 operational traces
conditions=[('90° / 90°',ROOT/'il_results/stage4/routed_eval/90',blue),
            ('270° / 270°',ROOT/'il_results/stage4/routed_eval/270',gold),
            ('240° / 240°',ROOT/'il_results/stage4/routed_eval/noncardinal_240_240_heading22',green)]
fig,axs=plt.subplots(2,2,figsize=(10.5,8.2),layout='constrained')
for title,path,color in conditions:
    if not(path/'trajectory.csv').is_file():continue
    d=pd.read_csv(path/'trajectory.csv');t=d.time_s/60
    axs[0,0].plot(d.x_m,d.y_m,label=title,color=color,lw=2)
    axs[0,1].plot(t,d.position_error_m,color=color,label=title,lw=1.8)
    axs[1,0].plot(t,d.heading_error_deg,color=color,label=title,lw=1.7)
    axs[1,1].plot(t,d.hull_gap_m,color=color,label=title,lw=1.7)
axs[0,0].scatter([50],[-100],s=100,marker='*',color=red,zorder=5,label='Target')
axs[0,0].set(xlabel='East x (m)',ylabel='North y (m)',title='(a) Recorded ship-centre trajectories');axs[0,0].axis('equal')
axs[0,1].set(xlabel='Simulated time (min)',ylabel='Position error (m)',title='(b) Terminal convergence')
axs[0,1].axhline(2.5,color=gray,ls='--',lw=.9,label='Acceptance threshold')
axs[1,0].set(xlabel='Simulated time (min)',ylabel='Heading error (deg)',title='(c) Heading regulation')
axs[1,0].axhline(2.,color=gray,ls='--',lw=.9)
axs[1,1].set(xlabel='Simulated time (min)',ylabel='Minimum hull–quay clearance (m)',title='(d) Geometric clearance')
for ax in axs.flat:ax.grid(alpha=.18)
axs[0,0].legend(frameon=False,loc='best')
save(fig,'Fig02_ThreeBerthingConditions')

# Figure 3: 270/270 ablation with exact recorded traces
fig,axs=plt.subplots(1,2,figsize=(10.7,3.95),layout='constrained')
experiments=[('Neural specialist only',ROOT/'il_results/stage4/candidate_eval/270',gray),
             ('Specialist + local progress',ROOT/'il_results/stage4/routed_eval/270',green)]
for name,path,color in experiments:
    if not(path/'trajectory.csv').is_file():continue
    d=pd.read_csv(path/'trajectory.csv');t=d.time_s/60
    axs[0].plot(t,d.position_error_m,lw=2,color=color,label=name)
    axs[1].plot(t,d.hull_gap_m,lw=2,color=color,label=name)
axs[0].axhline(2.5,color=red,lw=1,ls='--',label='Initial acceptance')
axs[0].set(xlabel='Simulated time (min)',ylabel='Position error (m)',title='(a) Residual terminal error')
axs[1].set(xlabel='Simulated time (min)',ylabel='Hull–quay clearance (m)',title='(b) Quay clearance')
for ax in axs:ax.grid(alpha=.2)
axs[0].legend(fontsize=8.0,loc='upper right',frameon=False)
save(fig,'Fig03_TerminalProgressAblation270')

# Figure 4: independent previously inspected noncardinal case matched controls
fig,axs=plt.subplots(1,2,figsize=(10.7,3.9),layout='constrained')
experiments=[('Stage 3 + local guard',ROOT/'il_results/stage4/matched_stage3_baseline/noncardinal_240_240_heading22',gray),
             ('Routed IL + guard + local trim',ROOT/'il_results/stage4/routed_eval/noncardinal_240_240_heading22',green)]
for name,path,color in experiments:
    if not(path/'trajectory.csv').is_file():continue
    d=pd.read_csv(path/'trajectory.csv');t=d.time_s/60
    axs[0].plot(d.x_m,d.y_m,color=color,lw=2,label=name)
    axs[1].plot(t,d.position_error_m,color=color,lw=2,label=name)
axs[0].scatter([50],[-100],marker='*',s=100,color=red,zorder=4)
axs[1].axhline(2.5,ls='--',lw=1,color=red)
axs[0].set(xlabel='East x (m)',ylabel='North y (m)',title='(a) Physical path');axs[0].axis('equal')
axs[1].set(xlabel='Simulated time (min)',ylabel='Position error (m)',title='(b) Error decay')
for ax in axs:ax.grid(alpha=.2)
axs[0].legend(frameon=False,loc='best',fontsize=8)
save(fig,'Fig04_NoncardinalComparison')

# Figure 6: local actuator fractions and safety interventions for adverse case
fig,axs=plt.subplots(2,1,figsize=(10.4,5.9),layout='constrained',sharex=True)
d=pd.read_csv(ROOT/'il_results/stage4/routed_eval/270/trajectory.csv');t=d.time_s/60
for i,col in enumerate([blue,green,gold,red],1):
    axs[0].plot(t,d[f'tug{i}_fraction'],color=col,lw=1.2,label=f'Tug {i}')
axs[1].plot(t,d.ship_speed_mps,color=blue,lw=1.7,label='Ship speed')
axs[1].plot(t,d.approach_speed_mps,color=red,lw=1.5,label='Quay-approach speed')
axs[0].set(ylabel='Local thrust fraction',title='(a) Independent tug commands')
axs[1].set(xlabel='Simulated time (min)',ylabel='Speed (m/s)',title='(b) Vessel speeds')
for ax in axs:ax.grid(alpha=.2);ax.legend(frameon=False,ncol=4)
save(fig,'Fig06_ActuationAndApproach270')
print('Created',len(list(OUT.glob('*.pdf'))),'vector figures in',OUT)

# Figure 7: all six frozen prospective cases; plot explicit failure statuses.
results_path=ROOT/'il_results/stage4/frozen_holdout_stage4/summary.json'
if results_path.exists():
    import json
    res=json.loads(results_path.read_text())
    names=[r['case'].split('_')[0] for r in res]
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.1),layout='constrained')
    colors=[green if r['status']=='SUCCESS' else red for r in res]
    y=np.arange(len(res))
    dur=np.array([r['seconds']/60 if r['seconds'] is not None else np.nan for r in res])
    err=np.array([r['position_error_m'] if r['position_error_m'] is not None else np.nan for r in res])
    axes[0].barh(y,dur,color=colors,alpha=.88,height=.62)
    axes[0].axvline(6500/60,ls='--',color=gray,lw=1)
    axes[0].set(yticks=y,yticklabels=names,xlabel='Simulated time at termination (min)',title='(a) Recorded duration')
    axes[0].invert_yaxis()
    axes[1].barh(y,err,color=colors,alpha=.88,height=.62)
    axes[1].axvline(2.5,ls='--',color=gray,lw=1,label='Acceptance limit')
    axes[1].set(yticks=y,yticklabels=names,xlabel='Final position error (m)',title='(b) At success or timeout')
    axes[1].invert_yaxis()
    for ax in axes:ax.grid(axis='x',alpha=.14)
    axes[1].legend(frameon=False,loc='lower right',fontsize=8)
    fig.suptitle('Prespecified environmental holdouts: teal = SUCCESS, red = TIMEOUT',fontsize=11.5,weight='bold')
    save(fig,'Fig07_FrozenHoldoutOutcomes')

# Figure 8: strictly paired frozen holdouts, no selective omission.
basefile=ROOT/'il_results/stage4/frozen_holdout_baseline/summary.json'
newfile=ROOT/'il_results/stage4/frozen_holdout_stage4/summary.json'
if basefile.exists() and newfile.exists():
    import json
    bas=json.loads(basefile.read_text());new=json.loads(newfile.read_text())
    lookup={(d['case']):d for d in new};keys=[d['case'] for d in bas]
    if set(keys)!=set(lookup):raise ValueError('Nonmatching frozen evaluation cases')
    fig,ax=plt.subplots(figsize=(10.8,4.15))
    fig.subplots_adjust(left=.065,right=.985,bottom=.265,top=.72)
    x=np.arange(len(keys));w=.34
    colors={'SUCCESS':green,'TIMEOUT':gold,'UNSAFE':red,'RISK_ABORT':red,'WALL_TIME_ABORT':gray,'PROCESS_TIMEOUT':gray}
    short={'SUCCESS':'OK','TIMEOUT':'TIME\nLIMIT','UNSAFE':'UNSAFE','RISK_ABORT':'RISK\nABORT','WALL_TIME_ABORT':'WATCHDOG','PROCESS_TIMEOUT':'WATCHDOG'}
    for j,series in enumerate([bas,new]):
        centers=x+(j-.5)*w
        ax.bar(centers,[1.]*len(keys),width=w,
               color=[colors.get(d['status'],gray) for d in series],alpha=.93,
               edgecolor='white',linewidth=1.3,zorder=3)
        for z,d in enumerate(series):
            ax.text(centers[z],.50,short.get(d['status'],d['status']),zorder=10,
                va='center',ha='center',color='white',fontsize=7.8,weight='bold')
    ax.set(xlim=(-.52,5.52),ylim=(0,1.05),xticks=x,
           xticklabels=[d['case'].split('_')[0] for d in bas],
           xlabel='Prespecified environmental holdout case')
    ax.set_yticks([])
    for edge in ['left','right','top']:ax.spines[edge].set_visible(False)
    ax.spines['bottom'].set_color('#c7d1da')
    ax.tick_params(axis='x',length=0,labelsize=11,pad=8)
    fig.text(.5,.955,'Paired environmental holdouts: recorded termination status',
             fontsize=12.5,weight='bold',ha='center',color=navy)
    fig.text(.5,.865,'Within each pair:  left = previous balanced IL   |   right = routed hybrid',
             ha='center',fontsize=10,color=navy)
    from matplotlib.patches import Patch
    legend=[Patch(color=green,label='First sustained acceptance'),
            Patch(color=gold,label='Simulation time limit'),
            Patch(color=red,label='Unsafe termination')]
    fig.legend(handles=legend,ncol=3,frameon=False,loc='upper center',
               bbox_to_anchor=(.5,.81),fontsize=8.9,handlelength=1.1,columnspacing=2.)
    fig.text(.5,.065,'Successful berthings: 2/6 to 4/6    |    Unsafe terminations: 1/6 to 0/6',
             ha='center',fontsize=10.8,color=navy,weight='bold')
    save(fig,'Fig08_PairedFrozenHoldouts')

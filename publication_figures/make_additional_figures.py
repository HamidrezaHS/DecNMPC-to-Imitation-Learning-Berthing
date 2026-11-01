from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
ROOT=Path('/mnt/data/IL_Publication_Stage4/project')
B=Path('/mnt/data/AOR_IL_Submission_Work/benchmark')
O=Path('/mnt/data/AOR_IL_Submission_Work/new_figures')
O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':12,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'savefig.dpi':270})
COL={'teacher':'#235b86','il':'#149883','highlight':'#bc654b','text':'#193346'}

def save(fig,name):
    fig.savefig(O/(name+'.pdf'),bbox_inches='tight',pad_inches=.13)
    fig.savefig(O/(name+'.png'),bbox_inches='tight',pad_inches=.13)
    plt.close(fig)

cases=[]
for wd,cd in [(90,90),(270,270)]:
    name=f'combined_psi20_w8_wd{wd}_c0p25_cd{cd}'
    t=json.loads((ROOT/'expert_archive/environment_4x4_v31'/name/'metrics.json').read_text())
    s=json.loads((ROOT/f'il_results/stage4/routed_eval/{wd}/metrics.json').read_text())
    cases.append((wd,t,s))

fig,axs=plt.subplots(1,2,figsize=(10.9,4.25),constrained_layout=True)
for ax,attr,scale,label in [(axs[0],'seconds',1/60,'Time to initial acceptance (min)'),(axs[1],'position_error_m',1.,'Position error at acceptance (m)')]:
    vals=np.array([[x[1][attr]*scale,x[2][attr]*scale] for x in cases]);x=np.arange(len(cases));w=.29
    r1=ax.bar(x-w/2,vals[:,0],w,label='Decentralized NMPC teacher',color=COL['teacher'],alpha=.9)
    r2=ax.bar(x+w/2,vals[:,1],w,label='Routed IL + local supervision',color=COL['il'],alpha=.9)
    for bars in [r1,r2]:
      for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height() + (.8 if attr=='seconds' else .03),f'{b.get_height():.2f}',ha='center',va='bottom',fontsize=9,color=COL['text'])
    ax.set_xticks(x, [r'90° / 90°', r'270° / 270°'])
    ax.set_ylabel(label);ax.set_xlabel('Wind / current toward-directions')
    ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(vals.flatten())*(1.16 if attr=='seconds' else 1.13))
    if attr=='position_error_m': ax.axhline(2.5,ls='--',color=COL['highlight'],lw=1.4,label='2.5 m acceptance threshold')
    ax.legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle('Matched closed-loop teacher–student development comparisons',fontsize=13,color=COL['text'],fontweight='bold')
save(fig,'Fig09_TeacherStudentMatched')

with (B/'decision_latency_samples.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
a=np.array([float(r['IL_cycle_ms']) for r in rows]);b=np.array([float(r['NMPC_cycle_ms']) for r in rows])
fig,axs=plt.subplots(1,2,figsize=(10.9,4.35),constrained_layout=True,width_ratios=(1.05,1.1))
ax=axs[0]
box=ax.boxplot([a,b],tick_labels=['Routed IL + local filter','Four independent NMPCs'],patch_artist=True,showfliers=True,medianprops=dict(color='white',lw=1.6),whiskerprops=dict(color='#718396'),capprops=dict(color='#718396'))
for p,c in zip(box['boxes'],[COL['il'],COL['teacher']]):p.set_facecolor(c);p.set_alpha(.9)
ax.set_ylabel('Four-agent decision-cycle time (ms)');ax.set_yscale('log');ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
ax.text(.02,.95,f'IL median = {np.median(a):.2f} ms\nNMPC median = {np.median(b):.2f} ms\nRatio of medians = {np.median(b)/np.median(a):.1f}×',transform=ax.transAxes,va='top',fontsize=9,weight='medium',color=COL['text'],bbox=dict(boxstyle='round,pad=0.5',facecolor='#f4f7f9',edgecolor='#d3dee5'))
ax=axs[1]
for wd,color in [(90,COL['teacher']),(270,COL['highlight'])]:
    ids=[i for i,r in enumerate(rows) if int(r['wind_to_deg'])==wd]
    ax.scatter(a[ids],b[ids],s=28,alpha=.80,color=color,label=f'{wd}° / {wd}°',edgecolors='white',linewidths=.25)
ax.set_xlabel('IL four-agent cycle (ms)');ax.set_ylabel('NMPC four-agent cycle (ms)')
ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=9)
fig.suptitle('Measured CPU latency on the same 36 teacher-generated ship–tug snapshots',fontsize=12.6,fontweight='bold',color=COL['text'])
save(fig,'Fig10_PairedLatencyBenchmark')

summary={'matched_scenarios':[{'wind_dir':wd,'current_dir':wd,'teacher':{'status':te['status'],'time_s':te['seconds'],'error_m':te['position_error_m'],'integrated_abs_thrust_Ns':te['integrated_abs_thrust_Ns']},'IL':{'status':il['status'],'time_s':il['seconds'],'error_m':il['position_error_m'],'integrated_abs_thrust_Ns':il['integrated_abs_thrust_Ns']}} for wd,te,il in cases], 'latency_summary':json.loads((B/'decision_latency_summary.json').read_text())}
(O/'matched_comparison_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Saved figs and summary:',O)

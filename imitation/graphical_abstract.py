"""Graphical abstract derived from actual simulation trajectories, not generated poses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,FancyBboxPatch,FancyArrowPatch
from plot_marl import _transform,_load_tug_outline
from visualization.marine_renderer import get_hull

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'manuscript_il'/'Submission_Files';OUT.mkdir(parents=True,exist_ok=True)
ship=np.asarray(get_hull(),float); tug=np.asarray(_load_tug_outline(),float)
navy='#16324d'; teal='#138d86';gold='#c98436';red='#b85958'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':10})
fig=plt.figure(figsize=(11.7,4.68),facecolor='white',layout='constrained')
gs=fig.add_gridspec(1,3,width_ratios=[1.1,1.15,1.0]);axs=[fig.add_subplot(gs[0,i]) for i in range(3)]
for ax in axs:ax.set_facecolor('#f4f8fa')
# A: logged ship/tug poses and trajectory
ax=axs[0]
d=pd.read_csv(ROOT/'il_results/stage4/routed_eval/270/trajectory.csv')
ax.plot(d.x_m,d.y_m,color=teal,lw=2.6)
for idx,alpha in [(0,.32),(-1,.92)]:
 r=d.iloc[idx]
 poly=_transform(ship,r.x_m,r.y_m,r.psi_deg)
 ax.add_patch(Polygon(poly,closed=True,facecolor='#b7c6d1',edgecolor=navy,lw=1.3,alpha=alpha,zorder=3))
 if idx==-1:
  for i,c in enumerate([teal,'#3b7eb0',gold,'#9459a5'],1):
   poly=_transform(tug,float(r[f'tug{i}_x_m']),float(r[f'tug{i}_y_m']),float(r[f'tug{i}_psi_deg']))
   ax.add_patch(Polygon(poly,closed=True,facecolor=c,edgecolor=navy,lw=.6,zorder=4))
ax.axhline(-133.5,color=gold,lw=3)
ax.text(-80,-140,'Quay',color=navy,fontsize=9,fontweight='bold')
ax.scatter([50],[-100],marker='*',color=red,s=80,zorder=5)
ax.set(xlim=(-230,230),ylim=(-165,150),title='Five-vessel berthing model',xlabel='East x (m)',ylabel='North y (m)')
ax.grid(alpha=.12)
# B: high-level process without text crowding
ax=axs[1];ax.set_xlim(0,10);ax.set_ylim(0,10);ax.axis('off')
def B(y,text,color):
 ax.add_patch(FancyBboxPatch((.65,y),8.7,1.45,boxstyle='round,pad=.10',facecolor=color,edgecolor=navy,lw=.6))
 ax.text(5,y+.73,text,ha='center',va='center',fontsize=10,fontweight='bold',color=navy)
def arrow(y):ax.annotate('',xy=(5,y-.25),xytext=(5,y+.35),arrowprops=dict(arrowstyle='-|>',color=navy,lw=1.5))
B(7.55,'Shared ship + own-tug state','#e8f0f7')
B(5.25,'Two trained local neural options','#d8e8f6')
B(2.95,'Environment-dependent routing','#f3eadb')
B(.65,'Local progress + safety supervision','#deeee9')
for y in [7.2,4.9,2.6]:arrow(y)
ax.set_title('Independent control at each tug',fontsize=11)
# C: factual tested outcome
ax=axs[2]
p=ROOT/'il_results/stage4/frozen_holdout_stage4/summary.json'
res=json.loads(p.read_text());labels=[r['case'][0] for r in res];y=np.arange(len(res))
colors=[teal if r['status']=='SUCCESS' else red for r in res]
ax.barh(y,np.ones(len(res)),color=colors,height=.64)
for k,r in enumerate(res):ax.text(.5,k,'SUCCESS' if r['status']=='SUCCESS' else 'TIMEOUT',ha='center',va='center',color='white',fontsize=9,fontweight='bold')
ax.set(yticks=y,yticklabels=labels,xlim=(0,1),title='Six predeclared perturbations')
ax.invert_yaxis();ax.set_xticks([])
for spine in ax.spines.values():spine.set_visible(False)
ax.text(.5,1.09,'4 / 6 entered initial acceptance',transform=ax.transAxes,ha='center',fontsize=9.6,color=navy,fontweight='bold')
ax.text(.5,-.11,'Numerical simulations; no safety guarantee',transform=ax.transAxes,ha='center',fontsize=8.2,color=red)
fig.savefig(OUT/'Graphical_Abstract_IL.png',dpi=300,bbox_inches='tight',pad_inches=.10)
fig.savefig(OUT/'Graphical_Abstract_IL.pdf',bbox_inches='tight',pad_inches=.10)
fig.savefig(OUT/'Graphical_Abstract_IL.tiff',dpi=300,bbox_inches='tight',pad_inches=.10)
plt.close(fig)
print('Graphical abstract saved:',OUT)

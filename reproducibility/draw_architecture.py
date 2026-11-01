"""Publication vector diagram for the decentralized IL actor architecture.
Creates Fig01_IndependentLearningArchitecture.pdf and matching PNG.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parent
NAVY = '#173850'
BLUE = '#2b668b'
TEAL = '#138a82'
ORANGE = '#c47b32'
SOFT_BLUE = '#edf5f9'
SOFT_TEAL = '#eaf6f3'
SOFT_ORANGE = '#fff4e8'
GRAY = '#637787'

fig, ax = plt.subplots(figsize=(13.4,4.65))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 13.4)
ax.set_ylim(0,4.65)
ax.axis('off')

def node(x,y,w,h,title,subtitle,edge,face,ts=12.4,ss=9.1):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.12',
                        facecolor=face,edgecolor=edge,lw=1.5,joinstyle='round')
    ax.add_patch(patch)
    ax.text(x+w/2,y+h*.68,title,color=NAVY,fontsize=ts,fontweight='bold',ha='center',va='center')
    ax.text(x+w/2,y+h*.31,subtitle,color=GRAY,fontsize=ss,ha='center',va='center',linespacing=1.4)
    return patch

def arrow(x0,y0,x1,y1,color='#637787',lw=1.7,scale=12):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=scale,
                                 color=color,lw=lw,connectionstyle='arc3'))

ax.text(0.18,4.37,'DECENTRALIZED IMITATION-LEARNING INFERENCE',fontsize=14.8,fontweight='bold',color=NAVY,ha='left')
ax.text(13.18,4.38,'Replicated independently for each tug  i = 1, ..., 4',fontsize=9.5,color=GRAY,ha='right')
ax.plot([.18,13.18],[4.17,4.17],color='#d9e5ea',lw=1.4)

node(.22,1.72,2.66,1.52,'Local observations',
     'Shared ship state + berth target\nOwn tug state\nLocal wind/current',BLUE,SOFT_BLUE,11.2,8.1)
node(3.44,2.47,2.56,1.05,'Standard policy','One trained actor per tug',BLUE,SOFT_BLUE,11.6,8.8)
node(3.44,1.15,2.56,1.05,'Adverse-flow policy','One trained specialist per tug',ORANGE,SOFT_ORANGE,11.6,8.8)
node(6.70,1.72,2.44,1.52,'Environment selector',
     'Wind/current projected\ntoward berth',TEAL,SOFT_TEAL,10.8,8.5)
node(9.65,1.72,3.38,1.52,'Local command processing',
     'Near-berth progress correction\n+ approach-speed filter',TEAL,SOFT_TEAL,10.9,8.6)

# Branches into two expert policies
ax.plot([2.88,3.1],[2.48,2.48],color=BLUE,lw=1.65)
ax.plot([3.10,3.10],[1.68,3.00],color=BLUE,lw=1.65)
arrow(3.10,3.00,3.44,3.00,color=BLUE)
arrow(3.10,1.68,3.44,1.68,color=ORANGE)
# Both candidate commands offered to gate
ax.plot([6.00,6.37],[3.00,3.00],color=BLUE,lw=1.55)
ax.plot([6.37,6.37],[2.48,3.00],color=BLUE,lw=1.55)
arrow(6.37,2.48,6.70,2.48,color=BLUE)
ax.plot([6.00,6.37],[1.68,1.68],color=ORANGE,lw=1.55)
ax.plot([6.37,6.37],[1.68,2.02],color=ORANGE,lw=1.55)
arrow(6.37,2.02,6.70,2.02,color=ORANGE)
arrow(9.14,2.48,9.65,2.48,color=TEAL)

# Outputs to shared plant below. Clear separation of inference and plant.
ax.plot([11.34,11.34],[1.72,1.38],color=TEAL,lw=1.6)
arrow(11.34,1.38,11.34,1.09,color=TEAL)
ax.text(11.34,.93,'Applied thrust + azimuth',fontsize=9.4,color=TEAL,fontweight='bold',ha='center')
ax.text(.22,.66,'FOUR INDEPENDENT LOCAL COMMANDS  →  COUPLED KVLCC2 + 4-TUG PHYSICAL SYSTEM',
        color=NAVY,fontsize=11.1,fontweight='bold',ha='left')
ax.text(.22,.30,'Ship-state feedback is shared; there is no peer-tug state/command exchange and no online NMPC solve.',
        color=GRAY,fontsize=9.6,ha='left')

fig.savefig(OUT/'Fig01_IndependentLearningArchitecture.pdf',bbox_inches='tight',pad_inches=.08,facecolor='white')
fig.savefig(OUT/'Fig01_IndependentLearningArchitecture.png',dpi=240,bbox_inches='tight',pad_inches=.08,facecolor='white')
plt.close(fig)

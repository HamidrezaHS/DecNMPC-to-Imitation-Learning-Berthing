"""Archived teacher-action agreement of the four deployed neural specialists.

This is an OFFLINE teacher-command regression diagnostic, NOT evidence of
closed-loop safety or complete equivalence of the final hybrid controller.
The environment gate selects one trained neural expert per tug, and we
compare that network's unsupervised raw output before slew limiting, local
progress assistance and the heuristic safety filter with the archived
Dec-NMPC command on the same causally aligned pre-decision observation.

Requires PyTorch, numpy, matplotlib and the repository's imitation package.
Usage (at repository root):
  python -m reproducibility.evaluate_teacher_imitation --out benchmark/imitation_quality
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from imitation.context_routed import toward_load_score
from imitation.policy import IndependentActors

ROOT = Path(__file__).resolve().parents[1]


def get_predictions(X, agent, std, adverse):
    """Evaluate the same per-agent env. gate as in deployed routed policy."""
    P = np.empty((len(X), 2), np.float32)
    routed_adverse = np.zeros(len(X), bool)
    with torch.no_grad():
        for i in range(1,5):
            inds = np.flatnonzero(agent == i)
            is_adv = np.asarray([toward_load_score(x) > 2. for x in X[inds]])
            routed_adverse[inds] = is_adv
            for pol, m in [(std, ~is_adv), (adverse, is_adv)]:
                sub = inds[m]
                if not len(sub):
                    continue
                z = np.clip((X[sub] - pol.mean) / pol.std, -10, 10)
                raw = pol.actors[i-1](torch.from_numpy(z.astype('float32')).to(pol.device)).cpu().numpy()
                P[sub,0] = np.clip(raw[:,0],0.,.90 if i<3 else .65)
                P[sub,1] = raw[:,1]
    return P, routed_adverse


def scores(teacher, student):
    force_t = np.stack([teacher[:,0]*np.cos(np.deg2rad(teacher[:,1]*20)),
                        teacher[:,0]*np.sin(np.deg2rad(teacher[:,1]*20))],axis=-1)
    force_s = np.stack([student[:,0]*np.cos(np.deg2rad(student[:,1]*20)),
                        student[:,0]*np.sin(np.deg2rad(student[:,1]*20))],axis=-1)
    active = teacher[:,0]>=.06
    return dict(n=int(len(teacher)),
                fraction_MAE=float(np.mean(np.abs(student[:,0]-teacher[:,0]))),
                force_vector_MAE=float(np.mean(np.linalg.norm(force_s-force_t,axis=1))),
                active_azimuth_MAE_deg=float(np.mean(np.abs(student[active,1]-teacher[active,1])*20)) if active.any() else None,
                n_active=int(active.sum()),
                teacher_near_zero_rate=float(np.mean(teacher[:,0]<.02)),
                student_near_zero_rate=float(np.mean(student[:,0]<.02)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,default=ROOT/'il_results/stage4/development_dagger_dataset.npz')
    ap.add_argument('--standard',type=Path,default=ROOT/'il_results/stage3/actors_balanced')
    ap.add_argument('--adverse',type=Path,default=ROOT/'il_results/stage4/actors_paper_candidate')
    ap.add_argument('--out',type=Path,default=ROOT/'benchmark/imitation_quality')
    args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    arch=np.load(args.dataset,allow_pickle=False)
    X=arch['X'].astype('float32');Y=arch['Y'].astype('float32');agent=arch['agent'];split=arch['split']
    standard=IndependentActors(args.standard);adverse=IndependentActors(args.adverse)
    P,flags=get_predictions(X,agent,standard,adverse)
    report={'meaning':'Archived expert-action agreement, raw neural specialist outputs; not post-processed final actions or closed-loop generalization',
            'angle_note':'Azimuth MAE is conditioned on teacher fraction >=0.06 to avoid undefined near-zero thrust angles',
            'caveat':'Test is one held-aside recorded demonstration case. Subsequent policy-development decisions may have been informed by related evaluation; do not treat as a prospective independent study.',
            'dataset':str(args.dataset), 'splits':{}}
    rows=[]
    for s in ['train','val','test']:
        m=split==s
        summary=scores(Y[m],P[m]);summary['adverse_gate_rate']=float(flags[m].mean())
        summary['by_agent']={str(i):scores(Y[m & (agent==i)],P[m & (agent==i)]) for i in range(1,5)}
        report['splits'][s]=summary
        for i in range(1,5):
            sm=summary['by_agent'][str(i)]
            rows.append(dict(split=s,agent=i,**sm))
    with (out/'by_agent.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/'teacher_student_action_agreement.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    # Compact publication-quality figure: sample-aligned cross-method agreement.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':10,'axes.labelsize':9.5,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(9.8,3.55))
    colors=['#24719e','#278b83','#bb7d30','#8658ab']
    ax=axs[0]
    for i,c in enumerate(colors,1):
        m=(split=='test')&(agent==i)
        ax.scatter(Y[m,0],P[m,0],s=13,alpha=.42,color=c,label=('Push '+str(i)) if i<3 else ('Tow '+str(i)),edgecolors='none')
    ax.plot([0,.95],[0,.95],'--',color='#a44e5b',lw=1.3,label='Perfect agreement')
    ax.set(xlim=(-.02,.98),ylim=(-.02,.98),xlabel='Dec-NMPC commanded thrust fraction',ylabel='IL neural thrust fraction')
    ax.set_title('(a) Archived test case: 1,056 agent decisions',pad=8)
    ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left',frameon=False,ncol=1,fontsize=8)
    ax=axs[1];loc=np.arange(4);bw=.34
    for ind,s in enumerate(['val','test']):
        vals=[report['splits'][s]['by_agent'][str(i)]['force_vector_MAE'] for i in range(1,5)]
        bars=ax.bar(loc+(-.5+ind)*bw, vals,width=bw,color=['#8396a9','#278b83'][ind],label={'val':'Validation','test':'Archived test'}[s])
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,v+.001,f'{v:.3f}',ha='center',va='bottom',fontsize=7.4)
    ax.set_xticks(loc,['Push 1','Push 2','Tow 3','Tow 4'])
    ax.set_ylim(0,max([report['splits'][s]['by_agent'][str(i)]['force_vector_MAE'] for s in ['val','test'] for i in range(1,5)])*1.32)
    ax.set_ylabel('Mean force-vector command error (normalized)')
    ax.set_title('(b) Magnitude + azimuth agreement by tug',pad=8)
    ax.legend(loc='upper right',frameon=False,fontsize=8)
    ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Direct teacher-to-student command imitation (before local supervision)',fontweight='bold',fontsize=11.5,y=.985,color='#193346')
    fig.tight_layout(rect=(0,.0,1,.92),w_pad=2)
    fig.savefig(out/'Fig06_TeacherCommandImitation.pdf',bbox_inches='tight',pad_inches=.10)
    fig.savefig(out/'Fig06_TeacherCommandImitation.png',dpi=250,bbox_inches='tight',pad_inches=.10)
    plt.close(fig)
    print(json.dumps(report['splits']['test'],indent=2))
    return report

if __name__=='__main__':
    main()

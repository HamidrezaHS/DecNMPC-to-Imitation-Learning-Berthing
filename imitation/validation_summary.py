"""Audit exact stage-2 recorded runs and render a status scorecard.

This script never synthesizes metrics and marks missing runs explicitly.
"""
from __future__ import annotations
import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CASES=[
 ('Teacher 90/90','expert_archive/environment_4x4_v31/combined_psi20_w8_wd90_c0p25_cd90/metrics.json','teacher','Training-context reference'),
 ('Original BC 90/90','il_results/eval_bc_90_90/metrics.json','BC initial','Training-context test'),
 ('Terminal BC 90/90','il_results/stage2/neural_terminal_90/metrics.json','BC terminal','Training-context test'),
 ('Terminal BC heading 22','il_results/stage2/neural_terminal_psi22/metrics.json','BC terminal','Unseen heading interpolation'),
 ('Terminal BC heading 24','il_results/stage2/neural_terminal_psi24/metrics.json','BC terminal','Unseen heading interpolation'),
 ('Terminal BC magnitude 7.7/0.22','il_results/stage2/neural_terminal_mag/metrics.json','BC terminal','Unseen magnitude interpolation'),
 ('Terminal BC calm','il_results/stage2/neural_terminal_calm/metrics.json','BC terminal','Training-context control'),
 ('Terminal BC heldout 270/270','il_results/stage2/neural_terminal_holdout270/metrics.json','BC terminal','Original held-out case'),
 ('DAgger round 2 90/90','il_results/stage2/dagger_round2_combo/metrics.json','DAgger round 2','Training-context control'),
 ('DAgger round 2 heading 22','il_results/stage2/dagger_round2_psi22/metrics.json','DAgger round 2','Unseen heading interpolation'),
 ('DAgger round 2 calm','il_results/stage2/dagger_round2_calm/metrics.json','DAgger round 2','Training-context control'),
 ('DAgger round 2 heldout 270/270','il_results/stage2/dagger_round2_holdout270/metrics.json','DAgger round 2','Original held-out case'),
 ('DAgger round 2 heading 25','il_results/stage2/prospective_heading25/heading25/metrics.json','DAgger round 2','Additional prospective interpolation'),
 ('DAgger round 2 magnitude 7.9/0.23','il_results/stage2/prospective_magnitude_new/magnitude_new/metrics.json','DAgger round 2','Additional prospective interpolation'),
 ('Calm-focused DAgger calm','il_results/stage2/focus_calm/metrics.json','Rejected exploratory variant','Training-context control'),
]

def main():
    out=Path('il_results/stage2');out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,path,method,split in CASES:
        p=Path(path)
        if not p.is_file():raise FileNotFoundError(str(p))
        m=json.loads(p.read_text(encoding='utf-8'))
        rows.append(dict(run=name,method=method,setting=split,status=m['status'],
            success=bool(m['success']),unsafe=bool(m['unsafe']),
            seconds=round(float(m['seconds']),3),position_error_m=round(float(m['position_error_m']),4),
            heading_error_deg=round(float(m['heading_error_deg']),4),
            model='numerical five-vessel simulator',
            student_expert_queries=m.get('expert_labels_collected',''),
            source_file=path))
    assert len(rows)==len(CASES)
    with (out/'STAGE2_VALIDATION_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'STAGE2_VALIDATION_RESULTS.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(11.3,7.5),layout='constrained')
    yy=np.arange(len(rows))[::-1]
    color={'SUCCESS':'#168c70','UNSAFE':'#c45455','TIMEOUT':'#b38a48','TIME_GUARD':'#b38a48'}
    for i,(r,y) in enumerate(zip(rows,yy)):
        c=color.get(r['status'],'#8c949e')
        ax.barh(y,1,height=.73,color=c,alpha=.90)
        suffix=f"{r['seconds']:.0f} s | {r['position_error_m']:.2f} m" if r['status']=='SUCCESS' else f"{r['position_error_m']:.2f} m"
        ax.text(1.015,y,f"{r['status']:<7}   {suffix}",va='center',ha='left',fontsize=9,color='#21374b')
    ax.set_yticks(yy,labels=[r['run'] for r in rows],fontsize=9)
    ax.set_xlim(0,1.78);ax.set_ylim(-1.3,len(rows)-0.2);ax.set_xticks([]);ax.spines[:].set_visible(False)
    fig.suptitle('Recorded closed-loop outcomes (single seed; no broad reliability claim)',fontsize=12,color='#273d51',weight='bold')
    ax.text(0,-0.95,'Green: terminal envelope satisfied | Red: unsafe stop | Ochre: time limit.  All runs are numerical simulations.',fontsize=9,color='#586a7d')
    fig.savefig(out/'stage2_outcome_scorecard.png',dpi=210,bbox_inches='tight')
    fig.savefig(out/'stage2_outcome_scorecard.pdf',bbox_inches='tight')
    plt.close(fig)
    print('Wrote',len(rows),'traceable run records and scorecard')
    return rows
if __name__=='__main__':main()

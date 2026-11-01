"""Create an honest figure/table comparing independently executed controllers.

This module does not invent missing test cases or compare unmatched missions.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(paths,out):
    cases=[];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for item in paths:
        item=Path(item);metrics=json.loads((item/'metrics.json').read_text())
        cases.append(dict(name=item.name,status=metrics['status'],success=metrics['success'],
                          unsafe=metrics['unsafe'],time_s=metrics['seconds'],
                          position_error_m=metrics['position_error_m'],
                          heading_error_deg=metrics['heading_error_deg'],
                          integrated_abs_thrust_Ns=metrics['integrated_abs_thrust_Ns']))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    (out/'comparison.json').write_text(json.dumps(cases,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(9,3.9),layout='constrained')
    names=[c['name'] for c in cases]
    axes[0].bar(names,[c['position_error_m'] for c in cases]);axes[0].axhline(2.5,ls='--',color='black',lw=1,label='2.5 m acceptance limit')
    axes[0].set(ylabel='Terminal position error (m)',title='Position at termination');axes[0].legend(frameon=False,fontsize=7)
    axes[1].bar(names,[c['time_s']/60 for c in cases]);axes[1].set(ylabel='Simulated time (min)',title='Duration until termination')
    for ax in axes:ax.tick_params(axis='x',labelrotation=45,labelsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'comparison.png',dpi=240,bbox_inches='tight');fig.savefig(out/'comparison.pdf',bbox_inches='tight');plt.close(fig)
    print(f'Saved report for {len(cases)} actual evaluations in {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cases',nargs='+');p.add_argument('--out',default='il_results/report')
    a=p.parse_args();report(a.cases,a.out)
if __name__=='__main__':main()

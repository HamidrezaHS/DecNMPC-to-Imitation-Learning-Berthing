"""Convert REAL recorded experimental outcomes into a self-contained TeX table.

This utility does not invent missing or teacher-unsupported entries. Do not
include a generated table in a paper without inspecting all process logs.
"""
from __future__ import annotations
import argparse,collections
from pathlib import Path
from .report import read_records,safe_num


def latex_esc(t):
    return str(t).replace('\\','\\textbackslash{}').replace('_','\\_').replace('%','\\%').replace('&','\\&')


def make_table(records):
    groups=collections.defaultdict(list)
    for r in records:groups[(r['method'],r['task'])].append(r)
    lines=[r'\begin{table}[t]',r'\centering\small',
           r'\caption{Prospective matched-task evaluations. Every attempted seed is included; blanks indicate unavailable metrics. These are observed outcomes, not an estimate of certified safety.}',
           r'\label{tab:prospective_evaluation}',r'\begin{tabular}{llrrrl}',r'\toprule',
           r'Method & Task & Success & Unsafe & Incomplete & Median successful time (s)\\',r'\midrule']
    for (method,task),rows in sorted(groups.items()):
        ok=[r for r in rows if r['success'] is True and r['status']=='SUCCESS']
        unsafe=sum(r['status']=='UNSAFE' for r in rows)
        inc=sum(r['status'] in ('UNSUPPORTED_COMPARISON','PROCESS_TIMEOUT','PROCESS_ERROR') for r in rows)
        valid=sorted(float(r['time_s']) for r in ok if safe_num(r['time_s']) is not None)
        med='--' if not valid else f'{(valid[len(valid)//2] if len(valid)%2 else (valid[len(valid)//2-1]+valid[len(valid)//2])/2):.0f}'
        lines.append(f'{latex_esc(method)} & {latex_esc(task)} & {len(ok)}/{len(rows)} & {unsafe} & {inc} & {med}'+r'\\')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return '\n'.join(lines)+'\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();rows=read_records(a.results)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(make_table(rows))
    print('Written:',dest)
if __name__=='__main__':main()

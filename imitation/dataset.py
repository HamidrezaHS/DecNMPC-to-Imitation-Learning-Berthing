"""Convert archived recorded trajectories and teacher decisions to causal samples.

Anti-leakage invariant: decision with observation_time_s=t belongs to the
trajectory row at t, not to the row after the physical simulation interval.
The exported `t_s` in decision logs is AFTER the interval and must NOT be
joined to state rows with the same t_s.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from .features import from_record, N_FEATURES, FEATURE_NAMES

DEFAULT_TEST={'v31_heading40_combo/combined_psi40_w8_wd90_c0p25_cd90',
              'environment_4x4_v31/combined_psi20_w8_wd270_c0p25_cd270'}
DEFAULT_VAL={'environment_4x4_v31/combined_psi20_w8_wd90_c0p25_cd270',
             'environment_4x4_v31/combined_psi20_w8_wd0_c0p25_cd180',
             'v31_regression_suite/deep200'}


def readcsv(path):
    with path.open(newline='',encoding='utf-8') as fp:return list(csv.DictReader(fp))


def collect_case(folder,root):
    name=str(folder.relative_to(root)).replace('\\','/')
    a=readcsv(folder/'trajectory.csv')
    d=readcsv(folder/'local_nmpc_decisions.csv')
    cfg=json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
    metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    if not bool(metrics['success']) or bool(metrics['unsafe']):
        raise ValueError(f'Non-success or unsafe teacher case: {name}')
    if len(d)!=4*(len(a)-1):
        raise ValueError(f'Log length mismatch: {name}, {len(a)} trajectory vs {len(d)} decision rows')
    goal=np.array([cfg['target_x'],cfg['target_y'],np.deg2rad(cfg['target_psi_deg'])])
    features=[];targets=[];agents=[];times=[]
    for k in range(len(a)-1):
        for i in range(1,5):
            dec=d[k*4+i-1]
            if int(dec['agent'])!=i:
                raise ValueError(f'Unexpected agent in case {name} at decision {k}: {dec["agent"]}')
            # This is the critical pre-decision time, not the recorded decision end time.
            err=abs(float(dec['observation_time_s'])-float(a[k]['time_s']))
            if err>0.05:
                raise ValueError(f'Decision/observation time mismatch {name}:{k}, {err=}')
            if abs(float(dec['t_s'])-float(a[k+1]['time_s']))>0.05:
                raise ValueError(f'Decision action and following physical step mismatch: {name}:{k}')
            feat=from_record(a[k],a[k-1] if k else None,i,goal)
            target=np.array([float(dec['fraction']),float(dec['offset_deg'])/20.],np.float32)
            if not np.isfinite(target).all() or not (-1.001<=target[1]<=1.001) or not (0<=target[0]<=1.001):
                raise ValueError(f'Invalid teacher action in {name}:{k}:{i}')
            features.append(feat);targets.append(target);agents.append(i);times.append(float(a[k]['time_s']))
    return (np.asarray(features,dtype=np.float32),np.asarray(targets,dtype=np.float32),
            np.asarray(agents,dtype=np.int32),np.asarray(times,dtype=np.float32),name)


def build_archive(archive,out,verbose=True):
    archive=Path(archive);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    dirs=sorted(p.parent for p in archive.glob('**/local_nmpc_decisions.csv'))
    if not dirs:raise FileNotFoundError('No archived teacher decisions found')
    feat=[];targets=[];agents=[];times=[];groups=[];splits=[]
    cases=[]
    for p in dirs:
        x,y,a,t,name=collect_case(p,archive)
        split='test' if name in DEFAULT_TEST else 'val' if name in DEFAULT_VAL else 'train'
        feat.append(x);targets.append(y);agents.append(a);times.append(t)
        groups.extend([name]*len(a));splits.extend([split]*len(a))
        cases.append(dict(case=name,split=split,decision_samples=len(a)//4,success=True))
        if verbose:print(f'{split:5s}  {len(a)//4:4d} decisions / {len(a):5d} agent samples  {name}',flush=True)
    x=np.concatenate(feat);y=np.concatenate(targets);a=np.concatenate(agents);t=np.concatenate(times)
    np.savez_compressed(out,X=x,Y=y,agent=a,t_s=t,case=np.array(groups,dtype=str),split=np.array(splits,dtype=str))
    manifest={'source':'archived teacher decisions and pre-decision trajectory rows',
              'feature_names':list(FEATURE_NAMES),'action_names':['fraction','offset_deg_over_20'],
              'feature_note':'Only measured own and shared ship variables; own velocity finite differences from history; no raw actuator azimuth in archived CSV.',
              'predecision_alignment':'log observation_time_s matches trajectory[k], action t_s matches trajectory[k+1]',
              'cases':cases,'samples':int(len(x)),'train_samples':int(sum(s=='train' for s in splits)),
              'val_samples':int(sum(s=='val' for s in splits)),
              'test_samples':int(sum(s=='test' for s in splits))}
    out.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',default='expert_archive');p.add_argument('--out',default='il_results/expert_dataset.npz')
    c=p.parse_args();print(json.dumps(build_archive(c.archive,c.out),indent=2))
if __name__=='__main__':main()

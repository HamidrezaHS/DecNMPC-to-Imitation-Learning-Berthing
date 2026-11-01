"""Export causally ordered local experiences from one IL rollout.

Each tug gets only its *own* current and past values plus the common ship
measurements. Ground truth task parameters are metadata of the evaluator and
must never be placed into policy observation/context at deployment.

Output can seed a recurrent context encoder in a future meta-RL extension;
this is NOT meta-RL training, adaptation or system identification itself.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np

SHIP=("x_m","y_m","psi_deg","u_mps","v_mps","r_deg_s")

def read_csv(path):
    with Path(path).open(newline='',encoding='utf8') as fp:return list(csv.DictReader(fp))

def v(row,key):
    a=row.get(key,'')
    return float(a) if a not in ('',None) else float('nan')


def export_episode(run_folder,out_folder):
    run_folder=Path(run_folder);out_folder=Path(out_folder);out_folder.mkdir(parents=True,exist_ok=True)
    rows=read_csv(run_folder/'trajectory.csv')
    if len(rows)<2:raise ValueError('At least two measured ship states required')
    times=[v(r,'time_s') for r in rows]
    if not all(math.isfinite(x) for x in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Nonmonotonic or nonfinite telemetry time')
    # A residual is calculated only *after* observing the next ship state,
    # never from future observations at policy decision instant t.
    paths=[]
    for agent in range(1,5):
        sequences=[]
        for k in range(len(rows)-1):
            a,b=rows[k:k+2]
            dt=v(b,'time_s')-v(a,'time_s')
            current=np.array([v(a,z) for z in SHIP],dtype=float)
            following=np.array([v(b,z) for z in SHIP],dtype=float)
            own=np.array([v(a,f'tug{agent}_{q}') for q in ('x_m','y_m','psi_deg','thrust_N')])
            action=np.array([v(a,f'tug{agent}_{q}') for q in ('fraction','offset_deg')])
            if not all(np.isfinite(x).all() for x in [current,following,own,action]):
                raise ValueError(f'Missing telemetry agent {agent} row {k}; do not impute')
            dv=(following[3:6]-current[3:6])/dt
            sequences.append(dict(t_observation_s=float(times[k]),t_next_observation_s=float(times[k+1]),dt_s=dt,
                   shared_ship_measurement=current.tolist(),private_own_tug=own.tolist(),
                   previous_applied_command=action.tolist(),
                   observed_next_ship_velocity=following[3:6].tolist(),
                   a_posteriori_ship_acceleration=dv.tolist(),
                   label_visibility='a_posteriori_ship_acceleration_only_after_next_measurement'))
        output=out_folder/f'agent_{agent}_local_history.jsonl'
        with output.open('w',encoding='utf8') as f:
            for q in sequences:f.write(json.dumps(q)+'\n')
        paths.append(str(output))
    (out_folder/'INTERFACE_DESCRIPTION.md').write_text(
        '# Offline local history interface for future meta-RL\n\n'
        'Four separate JSONL streams. Each line contains own-tug state, '
        'shared measured ship state, own previously applied command and '
        'the observed subsequent ship response. The last field is '
        '**a posteriori**, not a permissible input before the next observation. '
        'Do not expose hidden physics parameters or other tug states to the actor. '
        'This export is not a trained context inference module or a meta-RL result.\n',encoding='utf8')
    return paths


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print('\n'.join(export_episode(a.run,a.out)))
if __name__=='__main__':main()

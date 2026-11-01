"""Publication-development supervised distillation after train-only targeted DAgger.

- Four independent actors unchanged from the baseline architecture.
- Each trajectory receives equal *expected* sampler mass, regardless of duration.
- Terminal states are preferentially sampled; sample weights are NOT applied twice.
- Angle errors at zero thrust contribute little; physically relevant force vectors
  are compared to make errors more meaningful than normalized angle MSE alone.
- Data augmentation is restricted to measurement noise on existing causal inputs.
- Model selection uses prespecified whole-trajectory validation cases only.
This trainer does not assert closed-loop success or guaranteed safety.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from .features import FEATURE_NAMES
from .policy import TugActor


def sampler_weights(X, case, agent, split, terminal_gain=3.5):
    X=np.asarray(X);case=np.asarray(case);agent=np.asarray(agent);split=np.asarray(split)
    if not(len(X)==len(case)==len(agent)==len(split)):
        raise ValueError('array length mismatch')
    out=np.zeros(len(X),np.float64)
    # each distinct training case has equal expected sampling mass per agent
    dist=np.linalg.norm(X[:,:2].astype(float),axis=1)*200.
    for i in range(1,5):
        ids=np.flatnonzero((split=='train')&(agent==i))
        if len(ids)==0:raise ValueError(f'No train cases for agent {i}')
        for name in np.unique(case[ids]):
            loc=ids[case[ids]==name]
            terminal=1.+1.8*(dist[loc]<35.)+terminal_gain*(dist[loc]<12.)
            out[loc]=terminal/terminal.sum() # one trajectory one unit mass
    return out


def loss_function(p,y):
    pforce=p[:,0].clamp(0,1);yforce=y[:,0].clamp(0,1)
    # normalized teacher/student azimuth fractions mapped to actual angle degrees
    pa=p[:,1]*torch.pi/9.0;ya=y[:,1]*torch.pi/9.0
    vpx=pforce*torch.cos(pa);vpy=pforce*torch.sin(pa)
    vyx=yforce*torch.cos(ya);vyy=yforce*torch.sin(ya)
    zero=(yforce<.015).to(y.dtype)
    force=((pforce-yforce)**2)*(1+3*zero)
    vec=(vpx-vyx)**2+(vpy-vyy)**2
    angle=(p[:,1]-y[:,1])**2*(.06+.65*yforce)
    return (force+.5*vec+.08*angle).mean()


def train(dataset,out,base=None,epochs=80,patience=17,seed=143,device='auto',lr=.00022,
          batch=256,width=128,aug_std=.012):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if epochs<=0 or patience<=0:raise ValueError('Positive epoch and patience required')
    np.random.seed(seed);torch.manual_seed(seed);random.seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    with np.load(dataset) as d:
        X=d['X'].astype(np.float32);Y=d['Y'].astype(np.float32)
        case=d['case'];agent=d['agent'];split=d['split']
    tr=split=='train';va=split=='val';te=split=='test'
    if not(tr.any() and va.any() and te.any()):raise ValueError('All whole-case partitions required')
    if not(np.isfinite(X).all() and np.isfinite(Y).all()):raise ValueError('Nonfinite data')
    for name in np.unique(case):
        if len(np.unique(split[case==name]))!=1:raise ValueError('A case crosses partitions: '+str(name))
    base=Path(base) if base else None
    if base is not None:
        meta=json.loads((base/'actor_metadata.json').read_text())
        if meta['feature_names']!=list(FEATURE_NAMES):raise ValueError('Incompatible feature schema')
        mu=np.asarray(meta['mean'],np.float32);sigma=np.asarray(meta['std'],np.float32)
        width=int(meta['width'])
    else:
        mu=X[tr].mean(0);sigma=np.maximum(X[tr].std(0),.05)
    Z=np.clip((X-mu)/sigma,-10,10).astype(np.float32)
    weights=sampler_weights(X,case,agent,split)
    metadata=dict(feature_names=list(FEATURE_NAMES),mean=mu.tolist(),std=sigma.tolist(),width=width,
        method='train_only_case_balanced_dagger_distillation',seed=seed,
        train_cases=sorted(set(case[tr].tolist())),val_cases=sorted(set(case[va].tolist())),
        test_cases=sorted(set(case[te].tolist())),train_samples=int(tr.sum()),
        val_samples=int(va.sum()),test_samples=int(te.sum()),
        policy='four independent ship-state + own-tug local neural actors',
        source_dataset=str(dataset),notes='Fixed feature normalizer from train-only baseline; test labels never used in fitting')
    (out/'actor_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    log=[];scores=[]
    for i in range(1,5):
        tm=tr&(agent==i);vm=va&(agent==i);em=te&(agent==i)
        x=torch.from_numpy(Z[tm]);y=torch.from_numpy(Y[tm])
        sample_weight=torch.from_numpy(weights[tm])
        sampler=WeightedRandomSampler(sample_weight,num_samples=len(x),replacement=True)
        loader=DataLoader(TensorDataset(x,y),batch_size=batch,sampler=sampler)
        xv=torch.from_numpy(Z[vm]).to(device);yv=torch.from_numpy(Y[vm]).to(device)
        actor=TugActor(width).to(device)
        if base is not None:
            actor.load_state_dict(torch.load(base/f'actor_{i}.pt',map_location=device,weights_only=True))
        optimizer=torch.optim.AdamW(actor.parameters(),lr=lr,weight_decay=2e-4)
        best=np.inf;best_epoch=0;stale=0
        for ep in range(1,epochs+1):
            actor.train();total=0.;count=0
            for xb,yb in loader:
                xb=xb.to(device);yb=yb.to(device)
                if aug_std>0:
                    noise=torch.randn_like(xb)*aug_std
                    # No label-preserving pose perturbations. Only small sensor-noise-like nuisance.
                    mask=torch.zeros_like(noise);mask[:,11:14]=1;mask[:,15]=1;mask[:,18]=1
                    xb=xb+noise*mask
                optimizer.zero_grad(set_to_none=True)
                l=loss_function(actor(xb),yb)
                l.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),2.)
                optimizer.step();count+=len(xb);total+=len(xb)*float(l.item())
            actor.eval()
            with torch.no_grad():vl=float(loss_function(actor(xv),yv).item())
            log.append(dict(agent=i,epoch=ep,train_loss=total/max(count,1),val_loss=vl))
            if vl<best-1e-6:
                best=vl;stale=0;best_epoch=ep
                torch.save(actor.state_dict(),out/f'actor_{i}.pt')
            else:stale+=1
            if stale>=patience:break
        actor.load_state_dict(torch.load(out/f'actor_{i}.pt',map_location=device,weights_only=True))
        actor.eval()
        with torch.no_grad():p=actor(torch.from_numpy(Z[em]).to(device)).cpu().numpy()
        yy=Y[em]
        scores.append(dict(agent=i,best_epoch=best_epoch,val_loss=best,
            test_fraction_mae=float(np.abs(p[:,0]-yy[:,0]).mean()),
            test_offset_mae_deg=float((np.abs(p[:,1]-yy[:,1])*20).mean()),
            warning='Test label errors are descriptive only; closed-loop tests determine success'))
        print(f'[TRAIN] actor {i} epoch {best_epoch} val={best:.6f}',flush=True)
    with (out/'training_history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(log[0]));writer.writeheader();writer.writerows(log)
    (out/'training_metrics.json').write_text(json.dumps(scores,indent=2))
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    p.add_argument('--base',default=None);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--patience',type=int,default=17);p.add_argument('--seed',type=int,default=143)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--lr',type=float,default=.00022);p.add_argument('--aug-std',type=float,default=.012)
    a=p.parse_args()
    print(json.dumps(train(a.dataset,a.out,base=a.base,epochs=a.epochs,patience=a.patience,
                           seed=a.seed,device=a.device,lr=a.lr,aug_std=a.aug_std),indent=2))
if __name__=='__main__':main()

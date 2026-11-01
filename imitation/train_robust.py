"""Stage-3 multi-domain imitation: scenario-balanced risk-aware supervised training.

Strict train-only scaling and scenario balancing. Validation/test case rows and
labels are NEVER used in optimizer or fitting the OOD baseline. Does not imply
certified safety or closed-loop success; evaluate each checkpoint independently.
"""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from .features import FEATURE_NAMES
from .policy import TugActor


def training_weights(X, case, agent, split):
    """Positive per-sample weights: terminal, scenario, special environments.

    Within an agent, each trajectory contributes near-equal mass independent
    of its length. Target distance is a causal *pre-decision* measurement.
    """
    X=np.asarray(X);case=np.asarray(case);agent=np.asarray(agent);split=np.asarray(split)
    n=len(X)
    if not(len(case)==len(agent)==len(split)==n):raise ValueError('metadata lengths')
    distance=np.linalg.norm(X[:,:2],axis=1)*200.
    term=1.+1.5*(distance<50.)+3.*(distance<15.)
    weight=np.zeros(n,dtype=np.float64)
    train=np.where(split=='train')[0]
    for i in range(1,5):
        sub=train[agent[train]==i]
        names,counts=np.unique(case[sub],return_counts=True)
        if len(names)==0:raise ValueError('empty agent split')
        median=float(np.median(counts))
        length_factor={name:np.clip((median/c)**.7,.5,2.2) for name,c in zip(names,counts)}
        for idx in sub:
            weight[idx]=term[idx]*length_factor[case[idx]]
    return weight


def run_train(dataset,out,base=None,epochs=100,patience=18,seed=143,device='auto',batch=256,lr=3e-4,width=128):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    d=np.load(dataset);X=d['X'].astype(np.float32);Y=d['Y'].astype(np.float32)
    case=d['case'];split=d['split'];agent=d['agent']
    tr=split=='train';va=split=='val';te=split=='test'
    if not(tr.any() and va.any() and te.any()):raise ValueError('empty train/val/test')
    if not(np.isfinite(X).all() and np.isfinite(Y).all()):raise ValueError('nonfinite data')
    weights=training_weights(X,case,agent,split)
    base=Path(base) if base else None
    if base:
        meta=json.loads((base/'actor_metadata.json').read_text())
        if meta['feature_names']!=list(FEATURE_NAMES):raise ValueError('feature mismatch')
        mean=np.asarray(meta['mean'],np.float32);std=np.asarray(meta['std'],np.float32)
        width=int(meta['width'])
    else:
        mean=X[tr].mean(0);std=np.maximum(X[tr].std(0),.05)
    z=np.clip((X-mean)/std,-10,10).astype(np.float32)
    metadata=dict(feature_names=list(FEATURE_NAMES),mean=mean.tolist(),std=std.tolist(),width=width,
        method='scenario_balanced_terminal_weighted_imitation',seed=seed,
        train_cases=int(len(np.unique(case[tr]))),train_samples=int(tr.sum()),
        validation_samples=int(va.sum()),heldout_samples=int(te.sum()),
        split_rule='Fit only on train cases; held-out cases remain untouched',
        base_checkpoint=str(base) if base else None)
    (out/'actor_metadata.json').write_text(json.dumps(metadata,indent=2))
    results=[];history=[]
    for i in range(1,5):
        mtr=tr&(agent==i);mva=va&(agent==i);mte=te&(agent==i)
        actor=TugActor(width).to(device)
        if base:
            actor.load_state_dict(torch.load(base/f'actor_{i}.pt',map_location=device,weights_only=True))
        opt=torch.optim.AdamW(actor.parameters(),lr=lr,weight_decay=2e-4)
        xx=torch.from_numpy(z[mtr]);yy=torch.from_numpy(Y[mtr]);ww=torch.from_numpy(weights[mtr].astype(np.float32))
        sampler=WeightedRandomSampler(torch.from_numpy(weights[mtr].astype(np.float64)),num_samples=len(xx),replacement=True)
        loader=DataLoader(TensorDataset(xx,yy,ww),sampler=sampler,batch_size=batch)
        xv=torch.from_numpy(z[mva]).to(device);yv=torch.from_numpy(Y[mva]).to(device)
        best=float('inf');epochs_bad=0;best_epoch=0
        def loss(p,y,w=None):
            force=(p[:,0]-y[:,0]).square()*(1.+3.*(y[:,0]<.015).float())
            angle=(p[:,1]-y[:,1]).square()*(.1+.9*y[:,0])
            v=force+.40*angle
            return v.mean() if w is None else (v*w).sum()/w.sum()
        for epoch in range(1,epochs+1):
            actor.train();running=0.;seen=0
            for xb,yb,wb in loader:
                xb=xb.to(device);yb=yb.to(device);wb=wb.to(device)
                opt.zero_grad(set_to_none=True);p=actor(xb)
                l=loss(p,yb,wb);l.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),2.5);opt.step()
                running+=float(l.item())*len(xb);seen+=len(xb)
            actor.eval()
            with torch.no_grad():v=float(loss(actor(xv),yv).item())
            history.append(dict(agent=i,epoch=epoch,train_loss=running/seen,val_loss=v))
            if v<best-1e-6:
                best=v;best_epoch=epoch;epochs_bad=0;torch.save(actor.state_dict(),out/f'actor_{i}.pt')
            else:epochs_bad+=1
            if epoch==1 or epoch%10==0:
                print(f'[STAGE3] tug {i} epoch={epoch} train={running/seen:.5f} val={v:.5f} best={best:.5f}',flush=True)
            if epochs_bad>=patience:break
        actor.load_state_dict(torch.load(out/f'actor_{i}.pt',map_location=device,weights_only=True));actor.eval()
        with torch.no_grad():p=actor(torch.from_numpy(z[mte]).to(device)).cpu().numpy()
        y=Y[mte]
        results.append(dict(agent=i,best_epoch=best_epoch,val_loss=best,
            heldout_fraction_mae=float(np.abs(p[:,0]-y[:,0]).mean()),
            heldout_offset_mae_deg=float(20*np.abs(p[:,1]-y[:,1]).mean()),
            note='Open-loop heldout-label metrics; not evidence of closed-loop success'))
    with (out/'training_history.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['agent','epoch','train_loss','val_loss']);wr.writeheader();wr.writerows(history)
    (out/'training_metrics.json').write_text(json.dumps(results,indent=2))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='il_results/stage2/dagger_balanced/aggregated_round_02.npz')
    p.add_argument('--out',default='il_results/stage3/actors_balanced')
    p.add_argument('--base',default='il_results/stage2/dagger_balanced/actors_round_02')
    p.add_argument('--epochs',type=int,default=75);p.add_argument('--patience',type=int,default=14)
    p.add_argument('--seed',type=int,default=143);p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    p.add_argument('--lr',type=float,default=3e-4)
    a=p.parse_args();print(json.dumps(run_train(a.dataset,a.out,base=a.base,epochs=a.epochs,patience=a.patience,seed=a.seed,device=a.device,lr=a.lr),indent=2))
if __name__=='__main__':main()

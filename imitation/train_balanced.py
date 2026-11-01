"""Terminal-aware Behavior Cloning with emphasis on thrust-off accuracy.

Experimental variant: same actor architecture and strict whole-scenario split.
"""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from .features import FEATURE_NAMES
from .policy import TugActor


def run_train(dataset,out,epochs=100,batch_size=256,learning_rate=.001,width=128,seed=123,
              device='auto',patience=20,init_dir=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    data=np.load(dataset)
    X=data['X'].astype(np.float32);Y=data['Y'].astype(np.float32)
    # Normalize weight only from PRE-action target-distance, no future labels.
    distance=np.linalg.norm(X[:,:2],axis=1)*200.
    sample_weights=(1.+2.0*(distance<38.)+4.0*(distance<12.)).astype(np.float32)
    agents=data['agent'];splits=data['split']
    if not (len(X)==len(Y)==len(agents)==len(splits)):raise ValueError('Malformed training data')
    if not np.isfinite(X).all() or not np.isfinite(Y).all():raise ValueError('Nonfinite sample')
    train=splits=='train';val=splits=='val';test=splits=='test'
    if min(train.sum(),val.sum(),test.sum())==0:raise ValueError('Empty scenario split')
    # Global feature scaling is FIT ONLY on training scenarios (no validation/test leakage)
    mean=X[train].mean(0)
    std=np.maximum(X[train].std(0),.05).astype(np.float32)
    metadata=dict(feature_names=list(FEATURE_NAMES),mean=mean.tolist(),std=std.tolist(),
                  width=width,seed=seed,device=device,method='terminal_aware_Behavior_Cloning',
                  full_train_samples=int(train.sum()),val_samples=int(val.sum()),test_samples=int(test.sum()),
                  split_rule='whole trajectories split into train/val/test, no row-level leakage')
    (out/'actor_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    X=np.clip((X-mean)/std,-10.,10.).astype(np.float32)
    histories=[];metrics=[]
    for i in range(1,5):
        tr=train&(agents==i);va=val&(agents==i);te=test&(agents==i)
        actor=TugActor(width=width).to(device)
        if init_dir is not None:
            ckpt=Path(init_dir)/f'actor_{i}.pt'
            if ckpt.is_file():actor.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
        opt=torch.optim.AdamW(actor.parameters(),lr=learning_rate,weight_decay=.0001)
        def tensors(mask):
            return (torch.from_numpy(X[mask]).float(),torch.from_numpy(Y[mask]).float(),
                    torch.from_numpy(sample_weights[mask]).float())
        xt,yt,wt=tensors(tr);xv,yv,wv=tensors(va);xe,ye,we=tensors(te)
        loader=DataLoader(TensorDataset(xt,yt,wt),batch_size=batch_size,shuffle=True,drop_last=False)
        # offset loss fades toward low thrust: offset has little force meaning at zero thrust
        def loss_fn(pred,target,weights):
            # Near-zero tension commands must not turn into a persistent tow force.
            off=(target[:,0]<.015).float()
            loss_f=(pred[:,0]-target[:,0]).square()*(1.+3.*off)
            weight=.15+.85*target[:,0]
            loss_a=(pred[:,1]-target[:,1]).square()*weight
            return ((loss_f+.45*loss_a)*weights).sum()/weights.sum()
        best=float('inf');since=0;best_epoch=0
        for epoch in range(1,epochs+1):
            actor.train();train_loss=0.;seen=0
            for xb,yb,wb in loader:
                xb=xb.to(device);yb=yb.to(device);wb=wb.to(device)
                opt.zero_grad(set_to_none=True)
                pred=actor(xb);loss=loss_fn(pred,yb,wb)
                loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),3.)
                opt.step();train_loss+=float(loss.item())*len(xb);seen+=len(xb)
            actor.eval()
            with torch.no_grad():
                vl=float(loss_fn(actor(xv.to(device)),yv.to(device),wv.to(device)).item())
            train_loss/=max(1,seen)
            histories.append(dict(agent=i,epoch=epoch,train_loss=train_loss,val_loss=vl))
            if vl<best-1.e-6:
                best=vl;since=0;best_epoch=epoch
                torch.save(actor.state_dict(),out/f'actor_{i}.pt')
            else:since+=1
            if epoch==1 or epoch%10==0 or epoch==epochs:
                print(f'[BC] agent={i} epoch={epoch:4d} train={train_loss:.5f} val={vl:.5f} best={best:.5f}',flush=True)
            if since>=patience:break
        actor.load_state_dict(torch.load(out/f'actor_{i}.pt',map_location=device,weights_only=True));actor.eval()
        with torch.no_grad():
            p=actor(xe.to(device)).cpu().numpy()
        yte=ye.numpy()
        m=dict(agent=i,train_n=int(tr.sum()),val_n=int(va.sum()),test_n=int(te.sum()),
               best_epoch=best_epoch,val_loss=best,
               test_fraction_mae=float(np.abs(p[:,0]-yte[:,0]).mean()),
               test_offset_mae_deg=float((np.abs(p[:,1]-yte[:,1])*20.).mean()),
               note='Open-loop held-out imitation error; NOT a closed-loop berthing success rate')
        metrics.append(m)
    with (out/'history.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['agent','epoch','train_loss','val_loss']);w.writeheader();w.writerows(histories)
    (out/'bc_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    fig,axs=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for i,ax in enumerate(axs.flat,1):
        hist=[r for r in histories if r['agent']==i]
        ax.plot([r['epoch'] for r in hist],[r['train_loss'] for r in hist],label='Training',lw=1.5)
        ax.plot([r['epoch'] for r in hist],[r['val_loss'] for r in hist],label='Validation',lw=1.5)
        ax.set(title=f'Tug {i}: '+('Push' if i<3 else 'Tow'),xlabel='Epoch',ylabel='Weighted loss');ax.grid(alpha=.2)
        if i==1:ax.legend(frameon=False)
    fig.savefig(out/'bc_learning_curves.png',dpi=210,bbox_inches='tight')
    fig.savefig(out/'bc_learning_curves.pdf',bbox_inches='tight');plt.close(fig)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='il_results/expert_dataset.npz');p.add_argument('--out',default='il_results/bc')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--width',type=int,default=128)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--patience',type=int,default=20);p.add_argument('--init-dir',default=None)
    a=p.parse_args();print(json.dumps(run_train(a.dataset,a.out,a.epochs,a.batch_size,a.lr,a.width,a.seed,a.device,a.patience,a.init_dir),indent=2))
if __name__=='__main__':main()

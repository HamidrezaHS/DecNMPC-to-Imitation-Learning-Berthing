"""Local computer readiness check before expensive multi-seed experiments."""
from __future__ import annotations
import argparse,json,platform,shutil,sys
from pathlib import Path
from .runner import ROOT,load_protocol,checkpoint_digests,digest


def check(protocol):
    p=load_protocol(protocol)
    out=dict(platform=platform.platform(),python=sys.version.split()[0],
             cpu_threads=__import__('os').cpu_count(),free_disk_gb=round(shutil.disk_usage(ROOT).free/1e9,2),
             prospective_tasks=len(p['tasks']),seeds=p['seeds'],
             available_actor_checkpoints=checkpoint_digests('routed_il'),
             teacher_controller_digest=checkpoint_digests('teacher_nmpc'))
    try:
        import torch
        out['torch_version']=torch.__version__
        out['torch_cuda_available']=bool(torch.cuda.is_available())
        if torch.cuda.is_available():out['torch_gpu']=torch.cuda.get_device_name(0)
    except ImportError:out['torch_error']='PyTorch absent; see requirements-il.txt'
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',default='experiments/PROSPECTIVE_PROTOCOL_V1.json')
    a=p.parse_args();print(json.dumps(check(a.protocol),indent=2))
if __name__=='__main__':main()

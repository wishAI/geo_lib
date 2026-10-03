"""Provision only this sandbox; model downloads use immutable revisions."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from state import ROOT, OUT, REPO, write_json, sha256


def download_model():
    from huggingface_hub import snapshot_download
    pins=json.loads((ROOT/'provenance.json').read_text())
    pin=pins['model'];dest=OUT/'models'/pin['repo'].split('/')[-1]
    snapshot_download(repo_id=pin['repo'],revision=pin['revision'],local_dir=dest,
                      cache_dir=OUT/'hf_cache',allow_patterns=['config.yaml','model.safetensors','stats/**','LICENSE','README.md'])
    inventory=[]
    for p in sorted(dest.rglob('*')):
        if p.is_file() and '.cache' not in p.relative_to(dest).parts:
            inventory.append({'repoPath':str(p.relative_to(REPO)),'cloudPath':'remote_outputs/'+str(p.relative_to(REPO)),
                              'size':p.stat().st_size,'sha256':sha256(p),'requires_root_manifest':p.stat().st_size>5242880})
    write_json(OUT/'model_inventory.json',{'pin':pin,'files':inventory,'sync_status':'pending parent Nextcloud registration/sync'})
    print(dest)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['assets','model','status'])
    p.add_argument('--source',type=Path,default=REPO/'algorithms/urdf_learn_wasd_walk/inputs/landau_v10')
    args=p.parse_args()
    if args.command=='assets':
        from landau import copy_assets
        print(json.dumps(copy_assets(args.source),indent=2))
    elif args.command=='model':download_model()
    else:
        print((OUT/'backend_progress.json').read_text())

if __name__=='__main__':main()

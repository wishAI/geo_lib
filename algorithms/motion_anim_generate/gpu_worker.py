"""Allowlisted module for the parent's network-isolated motion worker."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import subprocess

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from state import ROOT, OUT, progress, write_json, sha256

PROMPTS={
    'idle':'A person stands still comfortably with their arms relaxed at their sides.',
    'walk':'A person walks forward in a straight line at a comfortable pace.',
    'turn':'A person turns ninety degrees to their left while staying in place.',
    'wave':'A person stands in place and waves their right hand above shoulder height.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--speed-bounded',action='store_true')
    p.add_argument('--unconditional',action='store_true');p.add_argument('--embeddings',type=Path)
    p.add_argument('--action',choices=list(PROMPTS),default='idle');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--seconds',type=float,default=2.);p.add_argument('--steps',type=int,default=30)
    p.add_argument('--run-id',default='unconditional_smoke_seed42');args=p.parse_args()
    occupancy=None
    if args.device=='cuda':
        # Query host driver occupancy before creating this process's CUDA context.
        occupancy=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        if occupancy.returncode or occupancy.stdout.strip():
            raise RuntimeError('GPU occupancy cannot be proven idle: '+(occupancy.stdout+occupancy.stderr).strip())
    import torch
    torch.set_num_threads(4)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('Worker CUDA Torch access failed; no hidden CPU fallback')
        torch.cuda.reset_peak_memory_stats()
    probe={'torch':torch.__version__,'cuda':torch.version.cuda,
           'device':torch.cuda.get_device_name(0) if args.device=='cuda' else 'explicit CPU',
           'matmul_sum':float((torch.ones((32,32),device=args.device)@torch.ones((32,32),device=args.device)).sum()),
           'text_encoder_device':os.environ.get('TEXT_ENCODER_DEVICE'),
           'pre_context_compute_processes':occupancy.stdout.strip() if occupancy else None,
           'host_gpu_idle_verified':occupancy is not None}
    write_json(OUT/('backend_gpu/torch_probe.json' if args.device=='cuda' else 'cpu_torch_probe.json'),probe)
    if args.probe:print(json.dumps(probe));return
    if not args.unconditional and not args.embeddings:
        raise RuntimeError('Need authorized exact-prompt embedding bundle; gated encoder is not bypassed')
    if args.embeddings:
        args.embeddings=args.embeddings.resolve()
        if not args.embeddings.is_relative_to(OUT.resolve()):raise ValueError('Embeddings must be task-local outputs')
        sidecar=args.embeddings.with_suffix('.json')
        provenance=json.loads(sidecar.read_text())
        pins=json.loads((ROOT/'provenance.json').read_text())
        if provenance.get('sha256')!=sha256(args.embeddings):raise ValueError('Embedding digest mismatch')
        for key in ('text_base','text_mntp','text_supervised'):
            if provenance.get(key)!=pins[key]:raise ValueError('Embedding encoder provenance mismatch: '+key)
        if not provenance.get('provenance') or not provenance.get('dtype'):raise ValueError('Missing embedding provenance/dtype')
    if not args.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run id')
    if not (1<=args.seconds<=10 and 1<=args.steps<=100):raise ValueError('Bounded runs: 1..10s and 1..100 steps')
    import numpy as np
    vendor=OUT/'vendor/kimodo'
    expected=json.loads((ROOT/'provenance.json').read_text())
    if subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()!=expected['kimodo']['revision']:
        raise ValueError('Unpinned vendor revision')
    os.environ['CHECKPOINT_DIR']=str(OUT/'models')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    sys.path.insert(0,str(vendor))
    from kimodo.model import load_model
    class Encoder:
        def __call__(self,texts):
            is_string=isinstance(texts,str);batch=[texts] if is_string else texts
            if args.unconditional:
                # This tensor is ignored by the actual unconditional CFG branch.
                # regular CFG weight=0 returns out_uncond exactly (cfg.py lines 73-93).
                values=torch.zeros((len(batch),1,4096));lengths=[1]*len(batch)
            else:
                with np.load(args.embeddings,allow_pickle=False) as bundle:
                    prompts=list(bundle['prompts'])
                    embedded=bundle['embeddings']
                    if embedded.shape!=(len(prompts),1,4096) or not np.isfinite(embedded).all():raise ValueError('Invalid embeddings shape/values')
                    values=torch.from_numpy(np.stack([bundle['embeddings'][prompts.index(t)] for t in batch])).float()
                lengths=[1]*len(batch)
            return (values[0],lengths[0]) if is_string else (values,lengths)
    run=OUT/'runs'/args.run_id;run.mkdir(parents=True,exist_ok=False)
    metadata={'source_kind':'local Kimodo unconditional inference' if args.unconditional else 'local Kimodo inference',
              'action':'unconditional' if args.unconditional else args.action,'prompt':None if args.unconditional else PROMPTS[args.action],
              'seed':args.seed,'seconds':args.seconds,'diffusion_steps':args.steps,'post_processing':False,
              'postprocessing_reason':'Raw diffusion feasibility first; compiled correction not installed',
              'cfg_type':'regular','cfg_weight':0. if args.unconditional else 2.,'pins':expected,
              'model_sha256':sha256(OUT/'models/Kimodo-SOMA-RP-v1.1/model.safetensors'),
              'embedding_sha256':sha256(args.embeddings) if args.embeddings else None,
              'command':sys.argv,'gpu':probe,'semantic_claim':not args.unconditional}
    write_json(run/'source_metadata.json',metadata)
    progress('local_inference',config=metadata,active_run=args.run_id,next_step='Retarget real diffusion output and validate; unconditional smoke cannot pass requested semantics')
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);np.random.seed(args.seed)
    start=time.monotonic()
    model=load_model('Kimodo-SOMA-RP-v1.1',device=args.device,text_encoder=Encoder())
    with torch.inference_mode():
        output=model('' if args.unconditional else PROMPTS[args.action],num_frames=round(args.seconds*model.fps),
                     num_denoising_steps=args.steps,cfg_type='regular',cfg_weight=metadata['cfg_weight'],
                     return_numpy=True,post_processing=False)
    np.savez_compressed(run/'source.npz',**output)
    metadata.update(elapsed_inference_s=time.monotonic()-start,source_sha256=sha256(run/'source.npz'),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else None)
    write_json(run/'source_metadata.json',metadata)
    del model;torch.cuda.empty_cache()
    from experiment import process
    result=process(run,'composite' if args.unconditional else args.action,metadata['source_kind'],args.speed_bounded)
    print(json.dumps({'inference':metadata,'validation_metrics':result['metrics']}))

if __name__=='__main__':main()

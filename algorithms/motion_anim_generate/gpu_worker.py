"""Allowlisted module for the parent's network-isolated motion worker."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

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
    p.add_argument('--unconditional',action='store_true');p.add_argument('--embeddings',type=Path)
    p.add_argument('--action',choices=list(PROMPTS),default='idle');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--seconds',type=float,default=2.);p.add_argument('--steps',type=int,default=30)
    p.add_argument('--run-id',default='unconditional_smoke_seed42');args=p.parse_args()
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Worker CUDA Torch access failed; no hidden CPU fallback')
    torch.cuda.reset_peak_memory_stats()
    probe={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),
           'matmul_sum':float((torch.ones((32,32),device='cuda')@torch.ones((32,32),device='cuda')).sum()),
           'text_encoder_device':os.environ.get('TEXT_ENCODER_DEVICE'),'network':'parent isolated worker'}
    write_json(OUT/'backend_gpu/torch_probe.json',probe)
    if args.probe:print(json.dumps(probe));return
    if not args.unconditional and not args.embeddings:
        raise RuntimeError('Need authorized exact-prompt embedding bundle; gated encoder is not bypassed')
    if not args.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run id')
    if not (1<=args.seconds<=10 and 1<=args.steps<=100):raise ValueError('Bounded runs: 1..10s and 1..100 steps')
    import numpy as np
    vendor=OUT/'vendor/kimodo'
    expected=json.loads((ROOT/'provenance.json').read_text())
    import subprocess
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
    model=load_model('Kimodo-SOMA-RP-v1.1',device='cuda',text_encoder=Encoder())
    with torch.inference_mode():
        output=model('' if args.unconditional else PROMPTS[args.action],num_frames=round(args.seconds*model.fps),
                     num_denoising_steps=args.steps,cfg_type='regular',cfg_weight=metadata['cfg_weight'],
                     return_numpy=True,post_processing=False)
    np.savez_compressed(run/'source.npz',**output)
    metadata.update(elapsed_inference_s=time.monotonic()-start,source_sha256=sha256(run/'source.npz'),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(run/'source_metadata.json',metadata)
    del model;torch.cuda.empty_cache()
    from experiment import process
    result=process(run,'composite' if args.unconditional else args.action,metadata['source_kind'])
    print(json.dumps({'inference':metadata,'validation_metrics':result['metrics']}))

if __name__=='__main__':main()

"""Official pinned encoder download/preparation; never an authentication workaround."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from state import ROOT,OUT,sha256,write_json,progress

KEYS=('text_base','text_mntp','text_supervised')
TEXT=OUT/'models/text'


def setup():
    destination=OUT/'encoder_venv'
    subprocess.run([sys.executable,'-m','venv',str(destination)],check=True)
    python=destination/'bin/python'
    env={**os.environ,'PIP_CACHE_DIR':str(OUT/'pip_cache')}
    subprocess.run([str(python),'-m','pip','install','--no-deps','-r',str(ROOT/'requirements-encoder.txt')],check=True,env=env)
    shared=list((OUT/'venv/lib').glob('python*/site-packages'))
    local=list((destination/'lib').glob('python*/site-packages'))
    if len(shared)!=1 or len(local)!=1:raise RuntimeError('Expected one task-local Python runtime')
    (local[0]/'task_runtime.pth').write_text(str(shared[0].resolve())+'\n')
    subprocess.run([str(python),str(ROOT/'encoder_check.py')],check=True,env={**env,'PYTHONDONTWRITEBYTECODE':'1'})


def download():
    # Parent invokes once AFTER approved-account confirmation. SDK credential is
    # read normally and passed in memory; never serialized or printed.
    from huggingface_hub import get_token,hf_hub_download,snapshot_download
    from huggingface_hub.errors import GatedRepoError
    token=get_token();pins=json.loads((ROOT/'provenance.json').read_text())
    try:
        hf_hub_download(pins['text_base']['repo'],'config.json',revision=pins['text_base']['revision'],
                        token=token,cache_dir=str(OUT/'encoder_hf_cache'))
    except GatedRepoError as error:
        write_json(OUT/'encoder_access.json',{'repo':pins['text_base']['repo'],'revision':pins['text_base']['revision'],
            'standard_sdk_credential_present':token is not None,'user_authorized_use':True,'status':'access_unavailable',
            'error_type':type(error).__name__,'http_status':getattr(error.response,'status_code',None)})
        raise SystemExit('Account access is not approved. Parent must obtain access before another attempt.') from None
    write_json(OUT/'encoder_access.json',{'repo':pins['text_base']['repo'],'revision':pins['text_base']['revision'],
        'standard_sdk_credential_present':token is not None,'user_authorized_use':True,'status':'access_verified'})
    inventory=[]
    for key in KEYS:
        patterns=['*.json','*.safetensors','*.model','LICENSE*','README.md']
        if key=='text_base':patterns=['config.json','generation_config.json','model*.safetensors','model.safetensors.index.json','tokenizer*','special_tokens_map.json','LICENSE*','README.md']
        snapshot_download(pins[key]['repo'],revision=pins[key]['revision'],token=token,local_dir=TEXT/key,
                          cache_dir=str(OUT/'encoder_hf_cache'),allow_patterns=patterns,max_workers=2)
        files=[]
        for path in sorted((TEXT/key).rglob('*')):
            if path.is_file() and '.cache' not in path.relative_to(TEXT/key).parts:
                files.append({'path':str(path.relative_to(OUT)),'size':path.stat().st_size,'sha256':sha256(path)})
        inventory.append({'key':key,'pin':pins[key],'files':files})
    write_json(OUT/'encoder_inventory.json',{'snapshots':inventory,'authentication':'standard HF SDK; no credential data stored'})
    write_json(OUT/'encoder_large_files.pending.json',{'version':1,'thresholdBytes':5242880,'status':'parent root registration/sync required',
        'files':[{'repoPath':'algorithms/motion_anim_generate/outputs/'+f['path'],
                  'cloudPath':'remote_outputs/algorithms/motion_anim_generate/outputs/'+f['path'],
                  'size':f['size'],'sha256':f['sha256']} for s in inventory for f in s['files'] if f['size']>5242880]})
    del token


def encode():
    if importlib.metadata.version('transformers')!='4.44.2':
        raise SystemExit('Use outputs/encoder_venv/bin/python: official bidirectional masking verified with Transformers4.44.2')
    pins=json.loads((ROOT/'provenance.json').read_text())
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=OUT/'vendor/kimodo',text=True).strip()!=pins['kimodo']['revision']:
        raise ValueError('Unpinned Kimodo source')
    inventory=json.loads((OUT/'encoder_inventory.json').read_text())
    for item in inventory['snapshots']:
        if item['pin']!=pins[item['key']]:raise ValueError('Encoder revision mismatch')
        for f in item['files']:
            if sha256(OUT/f['path'])!=f['sha256']:raise ValueError('Encoder file digest mismatch: '+f['path'])
    destination=OUT/'embeddings/official_prompts.npz'
    if destination.exists():raise FileExistsError('Preserve existing official embedding bundle; do not overwrite')
    derived=TEXT/'derived_mntp_local'
    if derived.exists():raise FileExistsError('Derived local MNTP already exists; inspect prior preparation')
    shutil.copytree(TEXT/'text_mntp',derived,ignore=shutil.ignore_patterns('.cache'))
    config=derived/'adapter_config.json';before=sha256(config);data=json.loads(config.read_text())
    data['base_model_name_or_path']=str((TEXT/'text_base').resolve());data['revision']=pins['text_base']['revision'];write_json(config,data)
    # Official prompt framing relies on this canonical name, not a local path.
    canonical=derived/'config.json';data=json.loads(canonical.read_text());data['_name_or_path']=pins['text_base']['repo'];write_json(canonical,data)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TEXT_ENCODER_DEVICE='cpu',
                      HF_HOME=str(OUT/'encoder_hf_cache'),CUDA_VISIBLE_DEVICES='')
    sys.path.insert(0,str(OUT/'vendor/kimodo'))
    import numpy as np
    import torch
    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
    from gpu_worker import PROMPTS
    torch.set_num_threads(4);start=time.monotonic()
    progress('official_encoder_cpu_embedding',active_process='official BF16 encoder, CPU, four prompts',next_step='Verify finite exact-prompt embeddings and publish bounded GPU requests')
    encoder=LLM2VecEncoder(str(derived),str(TEXT/'text_supervised'),'bfloat16',4096,device='cpu')
    texts=list(PROMPTS.values());embeddings,lengths=encoder(texts)
    if tuple(embeddings.shape)!=(4,1,4096) or not torch.isfinite(embeddings).all() or lengths!=[1]*4:
        raise ValueError('Unexpected official embeddings')
    destination.parent.mkdir(exist_ok=True);np.savez_compressed(destination,prompts=np.array(texts),embeddings=embeddings.float().cpu().numpy())
    import resource
    sidecar={key:pins[key] for key in KEYS}
    sidecar.update(sha256=sha256(destination),dtype='float32',compute_dtype='bfloat16 base; official PEFT adapters may be float32',
        provenance='Unmodified pinned official Kimodo LLM2VecEncoder, MNTP merge then supervised adapter; verified bidirectional-compatible runtime, batch_size=1 per upstream.',
        kimodo=pins['kimodo'],elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_inventory_sha256=sha256(OUT/'encoder_inventory.json'),local_path_rewrite={'original_adapter_config_sha256':before,'derived_sha256':sha256(config)},
        versions={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','tokenizers','accelerate','huggingface_hub']},
        prompts=PROMPTS,command=sys.argv)
    write_json(destination.with_suffix('.json'),sidecar)
    print(json.dumps({'embeddings':str(destination),'sha256':sidecar['sha256'],'shape':[4,1,4096]}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','download','encode']);a=p.parse_args()
    {'setup':setup,'download':download,'encode':encode}[a.action]()

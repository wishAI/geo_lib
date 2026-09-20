"""Sequential bounded unconditional baseline, never a substitute for requested prompt suite."""
import os
from pathlib import Path
import subprocess
import sys
from state import OUT, ROOT, write_json, progress


def main():
    results=[]
    for seed in (42,43,44):
        run=f'unconditional_cpu_6s_seed{seed}'
        command=[sys.executable,'-m','algorithms.motion_anim_generate.gpu_worker','--device','cpu',
                 '--unconditional','--speed-bounded','--seconds','6','--steps','100','--seed',str(seed),'--run-id',run]
        env={**os.environ,'TEXT_ENCODER_DEVICE':'cpu','HF_HOME':str(OUT/'hf_cache'),
             'XDG_CACHE_HOME':str(OUT/'cache'),'OPENBLAS_NUM_THREADS':'2'}
        progress('unconditional_multiseed',active_run=run,commands=[command],next_step='Review representative unconditional baseline; requested text semantics remain blocked')
        with (OUT/(run+'.log')).open('x') as log:
            try:result=subprocess.run(command,cwd=ROOT.parents[1],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=480)
            except subprocess.TimeoutExpired:code=124
            else:code=result.returncode
        results.append({'run_id':run,'seed':seed,'seconds':6,'steps':100,'exit_code':code,'command':command})
        write_json(OUT/'unconditional_suite.json',{'semantic_conditioning':False,'requested_prompt_suite':'blocked on authorized text encoder/embeddings','runs':results})
        if code:raise RuntimeError(f'Bounded experiment failed; inspect {run}.log')

if __name__=='__main__':main()

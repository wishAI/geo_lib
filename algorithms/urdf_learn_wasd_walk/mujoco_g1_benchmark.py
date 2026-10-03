#!/usr/bin/env python3
"""Bounded official G1 GPU throughput benchmark, never locomotion acceptance.

Outer runner uses stdlib only. --python selects the task-worker environment.
--dry-run neither probes GPU nor starts the selected Python. Two jobs run serially.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time

SCRIPT = Path(__file__).resolve()
ALGORITHM = SCRIPT.parent
ROOT = ALGORITHM.parents[1]
BACKEND = ALGORITHM / 'outputs/backend'
REPO = ROOT / 'helper_repos/unitree_rl_mjlab'
REVISION = '1425b15f73bd4095f0df53709d7c389c3eb9e790'
PINS = {'mjlab':'1.2.0', 'mujoco':'3.5.0', 'mujoco-warp':'3.5.0',
        'warp-lang':'1.12.0', 'rsl-rl-lib':'5.0.1', 'torch':'2.7.1', 'torchvision':'0.22.1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def worker(config_file):
    """Instrumentation changes timing only; runs unchanged official train.py."""
    beginning = time.perf_counter()
    cfg = json.loads(Path(config_file).read_text())
    directory = Path(cfg['directory'])
    events = directory/'events.jsonl'
    def event(kind, **values):
        append(events, {'event':kind, 'since_worker_start_s':time.perf_counter()-beginning, **values})
    event('worker_started', pid=os.getpid())
    versions = {p:importlib.metadata.version(p) for p in PINS}
    installed = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])
    save(directory/'packages.json', {'versions':versions, 'all_distributions':installed})
    wrong = {p:(version, PINS[p]) for p,version in versions.items() if version.split('+')[0] != PINS[p]}
    if wrong:
        raise RuntimeError(f'Package pin mismatch (actual, expected): {wrong}')
    import torch
    import warp as wp
    if not torch.cuda.is_available():
        raise RuntimeError('Task-scoped GPU is not accessible to selected Python; refusing CPU fallback')
    wp.config.kernel_cache_dir = str(directory/'warp_cache')
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    event('cuda_ready', torch_cuda_version=torch.version.cuda, device_name=torch.cuda.get_device_name(0),
          total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    from mjlab.sim.sim import Simulation
    from mjlab.envs import ManagerBasedRlEnv
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    from rsl_rl.algorithms import PPO
    from rsl_rl.utils.logger import Logger
    event('imports_complete')

    def timed_init(cls, name):
        original=cls.__init__
        def wrapped(self, *args, **kwargs):
            t=time.perf_counter(); event(name+'_begin')
            original(self, *args, **kwargs)
            torch.cuda.synchronize()
            event(name+'_end', duration_s=time.perf_counter()-t)
        cls.__init__=wrapped
    timed_init(Simulation, 'simulation_init_including_compile')
    timed_init(ManagerBasedRlEnv, 'environment_init_including_compile')
    timed_init(OnPolicyRunner, 'ppo_runner_init')
    original_learn, original_act = OnPolicyRunner.learn, PPO.act
    original_returns, original_log = PPO.compute_returns, Logger.log
    state={'active':False, 'act_count':0, 'rollout':24}
    def learn(self, *args, **kwargs):
        state.update(active=True, act_count=0, rollout=int(self.cfg['num_steps_per_env']))
        torch.cuda.synchronize(); event('learn_begin', rollout_steps=state['rollout'])
        try:
            return original_learn(self, *args, **kwargs)
        finally:
            torch.cuda.synchronize(); event('learn_end'); state['active']=False
    def act(self, *args, **kwargs):
        if state['active']:
            if state['act_count'] % state['rollout'] == 0:
                torch.cuda.synchronize()
                state['collection_start']=time.perf_counter()
                event('collection_begin', iteration=state['act_count']//state['rollout'])
            state['act_count']+=1
        return original_act(self, *args, **kwargs)
    def compute_returns(self, *args, **kwargs):
        if state['active']:
            torch.cuda.synchronize()
            boundary=time.perf_counter()
            state['collection_s']=boundary-state['collection_start']
            state['learning_start']=boundary
        return original_returns(self, *args, **kwargs)
    def log(self, *args, **kwargs):
        if state['active']:
            torch.cuda.synchronize()
            learning_s=time.perf_counter()-state['learning_start']
            collection_s=state['collection_s']
            item={'iteration':kwargs['it'], 'num_envs':self.num_envs, 'rollout_steps':state['rollout'],
                  'collection_s_synchronized':collection_s, 'learning_s_synchronized':learning_s,
                  'iteration_s_synchronized':collection_s+learning_s,
                  'native_collection_s':kwargs['collect_time'], 'native_learning_s':kwargs['learn_time'],
                  'control_transitions_per_s_collection':self.num_envs*state['rollout']/collection_s,
                  'control_transitions_per_s_total':self.num_envs*state['rollout']/(collection_s+learning_s),
                  'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                  'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                  'since_worker_start_s':time.perf_counter()-beginning}
            append(directory/'iterations.jsonl', item); event('iteration_complete', **item)
        return original_log(self, *args, **kwargs)
    OnPolicyRunner.learn, PPO.act, PPO.compute_returns, Logger.log = learn, act, compute_returns, log
    event('instrumentation_ready', explanation='CUDA synchronization only at rollout/learning boundaries; no per-step synchronization. Native timings retained separately.')
    sys.path.insert(0, str(REPO))
    sys.argv=cfg['official_argv']
    runpy.run_path(str(REPO/'scripts/train.py'),run_name='__main__')
    event('worker_completed')


def process_tree_stats(pid):
    """Read only this benchmark child and its descendants in this PID namespace."""
    pending=[pid]; seconds=rss=0; count=0
    ticks=os.sysconf('SC_CLK_TCK'); pagesize=os.sysconf('SC_PAGE_SIZE')
    while pending:
        p=pending.pop()
        try:
            stat=Path(f'/proc/{p}/stat').read_text().rsplit(')',1)[1].split()
            seconds+=(int(stat[11])+int(stat[12]))/ticks
            rss+=int(stat[21])*pagesize
            count+=1
            pending.extend(int(n) for n in Path(f'/proc/{p}/task/{p}/children').read_text().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return seconds,rss,count


def sample_resources(pid, directory, stop):
    last=time.perf_counter(); previous_cpu=0.
    while not stop.is_set():
        now=time.perf_counter(); cpu,rss,count=process_tree_stats(pid)
        row={'monotonic_s':now, 'pid':pid, 'process_tree_cpu_seconds':cpu,
             'process_tree_cpu_percent_one_core_100':max(0.,100*(cpu-previous_cpu)/max(now-last,1e-6)),
             'process_tree_rss_bytes':rss, 'process_count':count}
        # Initial CPU sample has no meaningful observation interval.
        if previous_cpu==0: row['process_tree_cpu_percent_one_core_100']=None
        try:
            mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in {'MemTotal','MemAvailable'}}
            row['host_memory']=mem
        except OSError as e: row['host_memory_error']=str(e)
        try:
            query=['nvidia-smi','--query-gpu=index,uuid,name,driver_version,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw','--format=csv,noheader,nounits']
            result=subprocess.run(query,capture_output=True,text=True,timeout=2)
            row['nvidia_smi']={'command':query,'returncode':result.returncode,'csv':result.stdout.strip(),'stderr':result.stderr.strip()}
            row['gpu_scope']='nvidia-smi reports all visible GPUs, including unrelated workloads; CUDA_VISIBLE_DEVICES is recorded in config'
        except (OSError,subprocess.TimeoutExpired) as e: row['nvidia_smi_error']=str(e)
        append(directory/'resources.jsonl',row)
        last,previous_cpu=now,cpu
        stop.wait(1.)


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []


def summarize(directory, config, command, wall_s, code, timed_out):
    iterations=read_jsonl(directory/'iterations.jsonl'); events=read_jsonl(directory/'events.jsonl')
    log=(directory/'train.log').read_text(errors='replace')
    # Warp prints measured module load/compilation times; distinguish cold compilation from cache reads.
    modules=[]
    for line in log.splitlines():
        m=re.search(r'Module (.*?) load on device .*? took ([0-9.]+) ms\s*(.*)',line)
        if m: modules.append({'module':m[1],'duration_s':float(m[2])/1000,'label':m[3].strip()})
    compiled_s=sum(m['duration_s'] for m in modules if 'compiled' in m['label'])
    cached_s=sum(m['duration_s'] for m in modules if 'cached' in m['label'])
    steady=iterations[1:]
    total_collection=sum(x['collection_s_synchronized'] for x in steady)
    total_learning=sum(x['learning_s_synchronized'] for x in steady)
    transitions=sum(x['num_envs']*x['rollout_steps'] for x in steady)
    configs={str(p.relative_to(directory)):sha(p) for p in directory.glob('logs/**/params/*.yaml')}
    checkpoints={str(p.relative_to(directory)):sha(p) for p in directory.glob('logs/**/*.pt')}
    begin=next((e['since_worker_start_s'] for e in events if e['event']=='learn_begin'),None)
    summary={'command':command,'official_command':config['official_argv'], 'env_overrides':config['env_overrides'],
             'wall_s':wall_s,'returncode':code,'timeout':timed_out,
             'positive_control_locomotion_pass':False,'training_convergence_claim':False,
             'num_envs':config['num_envs'],'requested_iterations':config['iterations'], 'completed_iterations':len(iterations),
             'startup_to_learn_s':begin,'compilation_module_s':compiled_s,'cached_module_load_s':cached_s,
             'module_timing_records':modules,'compile_timing_caveat':'Warp log module compilation durations are nested within startup or first rollout; do not add them to startup wall time. Other CUDA/JIT startup is not individually attributed.',
             'steady_definition':'All complete iterations after iteration0; 4env case therefore has one steady iteration.',
             'collection_s':[x['collection_s_synchronized'] for x in iterations],
             'learning_s':[x['learning_s_synchronized'] for x in iterations],
             'iteration_s':[x['iteration_s_synchronized'] for x in iterations],
             'control_transitions_per_s_steady':transitions/total_collection if total_collection else None,
             'control_transitions_per_s_total_steady':transitions/(total_collection+total_learning) if total_collection+total_learning else None,
             'physics_transitions_per_s_steady':4*transitions/total_collection if total_collection else None,
             'physics_decimation':4, 'actual_config_sha256':configs,'checkpoint_sha256':checkpoints,
             'config_sha256':sha(directory/'benchmark_config.json'),'source_sha256':sha(SCRIPT),
             'comparison_caveat':'Official G1 model and config. Compare matched4env case to CPU smoke; CPU native collection timings were not explicitly CUDA-synchronized. Larger batch tests scaling, not matched workload. No Landau performance or gate inference.'}
    save(directory/'result.json',summary)
    return summary


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        worker(sys.argv[2]); return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--timeout-s',type=float,default=300.)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if not 10<=args.timeout_s<=300: p.error('Each run deadline must be10..300seconds')
    output=args.output.resolve()
    if not output.is_relative_to(BACKEND.resolve()): p.error('Output must be inside this isolated algorithm outputs/backend')
    if args.dry_run:
        print(json.dumps({'python':str(args.python),'output':str(output),'sequential_runs':[{'num_envs':4,'iterations':2},{'num_envs':1024,'iterations':10}], 'per_run_timeout_s':args.timeout_s,'gpu_probe_or_training_executed':False},indent=2)); return
    python=args.python.expanduser().absolute()  # Preserve venv executable symlink; resolving loses its environment.
    if not python.is_file(): p.error('Selected Python does not exist')
    revision=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if revision!=REVISION: raise RuntimeError('Official helper repository revision mismatch')
    dirty=subprocess.run(['git','-C',str(REPO),'diff','--quiet','HEAD','--','scripts/train.py','src/tasks/velocity','src/assets/robots/unitree_g1'])
    if dirty.returncode: raise RuntimeError('Official training/G1 sources differ from pinned revision')
    output.mkdir(parents=True,exist_ok=False)
    source_files=[REPO/'scripts/train.py',REPO/'setup.py',REPO/'src/tasks/velocity/velocity_env_cfg.py']
    source_files+=list((REPO/'src/tasks/velocity/config/g1').glob('*.py'))
    source_files+=list((REPO/'src/assets/robots/unitree_g1').rglob('*'))
    source_hashes={str(f.relative_to(REPO)):sha(f) for f in source_files if f.is_file() and '__pycache__' not in f.parts}
    save(output/'provenance.json',{'revision':revision,'helper_source_sha256':source_hashes,'runner_sha256':sha(SCRIPT),'python':str(python),'required_pins':PINS,'created_at':datetime.now(timezone.utc).isoformat(),'no_landau_or_training_convergence_claim':True})
    (output/'runner_source.py').write_bytes(SCRIPT.read_bytes())
    results=[]
    for envs,iters in ((4,2),(1024,10)):
        directory=output/f'g1_gpu_env{envs}_iter{iters}_seed42'; directory.mkdir()
        for name in ('warp_cache','cache','torch_cache','cuda_cache','tmp','mpl','torch_extensions','triton','config'): (directory/name).mkdir()
        overrides={'MUJOCO_GL':'egl','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','WANDB_MODE':'disabled',
                   'PYTHONPATH':str(REPO),'WARP_CACHE_PATH':str(directory/'warp_cache'),'XDG_CACHE_HOME':str(directory/'cache'),
                   # NVRTC rejects very long TMPDIR names. The authorized worker
                   # provides a private writable /tmp; durable caches stay local.
                   'TORCH_HOME':str(directory/'torch_cache'),'CUDA_CACHE_PATH':str(directory/'cuda_cache'),'TMPDIR':'/tmp',
                   'MPLCONFIGDIR':str(directory/'mpl'),'TORCH_EXTENSIONS_DIR':str(directory/'torch_extensions'),
                   'TRITON_CACHE_DIR':str(directory/'triton'),'XDG_CONFIG_HOME':str(directory/'config')}
        # Preserve worker-assigned GPU restrictions; never enumerate/select a device outside them.
        if 'CUDA_VISIBLE_DEVICES' in os.environ: overrides['CUDA_VISIBLE_DEVICES']=os.environ['CUDA_VISIBLE_DEVICES']
        # The pinned CLI defaults to GPU list [0]. Passing scalar "0" to its
        # list/None/literal union is rejected by the installed tyro parser.
        official=[str(REPO/'scripts/train.py'),'Unitree-G1-Flat','--env.scene.num-envs',str(envs),
                  '--agent.seed','42','--agent.max-iterations',str(iters),'--agent.save-interval','1',
                  '--agent.logger','tensorboard','--agent.run-name',f'gpu_env{envs}_seed42']
        config={'directory':str(directory),'num_envs':envs,'iterations':iters,'seed':42,'env_overrides':overrides,
                'official_argv':official,'timeout_s':args.timeout_s,'cold_cache':True,'render':False}
        save(directory/'benchmark_config.json',config)
        command=[str(python),str(SCRIPT),'--worker',str(directory/'benchmark_config.json')]
        environment=os.environ.copy(); environment.update(overrides)
        beginning=time.perf_counter(); timed_out=False
        with (directory/'train.log').open('w') as log:
            process=subprocess.Popen(command,cwd=directory,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(directory/'active_process.json',{'pid':process.pid,'command':command,'shell_command':shlex.join(command),'started_at':datetime.now(timezone.utc).isoformat(),'timeout_s':args.timeout_s})
            stop=threading.Event(); sampler=threading.Thread(target=sample_resources,args=(process.pid,directory,stop),daemon=True); sampler.start()
            try:
                try: code=process.wait(timeout=max(0.1,args.timeout_s-2))
                except subprocess.TimeoutExpired:
                    timed_out=True
                    try: os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: code=process.wait(timeout=max(0.01,args.timeout_s-(time.perf_counter()-beginning)))
                    except subprocess.TimeoutExpired:
                        try: os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError: pass
                        code=process.wait()
            finally:
                if process.poll() is None:  # Includes interruption of this orchestrator.
                    try: os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
                process_wall=time.perf_counter()-beginning
                stop.set(); sampler.join(timeout=3)
        wall=time.perf_counter()-beginning
        result=summarize(directory,config,command,process_wall,code,timed_out)
        result['orchestration_wall_s_including_sampler_teardown']=wall
        save(directory/'result.json',result); results.append(result)
        save(directory/'active_process.json',{'pid':None,'completed':True,'returncode':code,'timeout':timed_out})
        save(output/'results.json',results)
        print(json.dumps({'run':str(directory),'returncode':code,'timeout':timed_out,'wall_s':wall,'steady_control_transitions_s':result['control_transitions_per_s_steady']}),flush=True)
        # Missing GPU/dependency failures will not improve at a larger batch. Cold compile timeouts
        # also do not justify launching a second larger compilation within this bounded request.
        if code!=0:
            raise RuntimeError(f'Official G1 child exited {code}; retained result at {directory}')


if __name__=='__main__':
    main()

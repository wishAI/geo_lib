"""Audited single-world GPU physics bridge for complete dynamics evaluation.

Every physics step is copied back with actual Warp contact/constraint forces.
This deliberately expensive evaluation bridge is not a training speed benchmark.
"""
from pathlib import Path
import importlib.metadata
import json
import os
import threading
import time

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from algorithms.urdf_learn_wasd_walk.mujoco_backend import digest, validate_physics_health, write_json
from algorithms.urdf_learn_wasd_walk.mujoco_g1_benchmark import sample_resources


class WarpEvaluation:
    def __init__(self, model, data, directory):
        try:
            self._initialize(model,data,directory)
        except Exception as error:
            if hasattr(self,'stop'):self.stop.set();self.sampler.join(timeout=3)
            write_json(directory/'warp_initialization_failure.json',{'error':repr(error),'source_sha256':digest(__file__)})
            raise

    def _initialize(self, model, data, directory):
        self.directory=directory
        (directory/'warp_runtime_source.py').write_text(Path(__file__).read_text())
        if model.opt.noslip_iterations:
            raise ValueError('Warp cannot implement noslip; a new explicit physics configuration is required')
        versions={p:importlib.metadata.version(p) for p in ('mujoco','mujoco-warp','warp-lang')}
        if versions != {'mujoco':'3.5.0','mujoco-warp':'3.5.0','warp-lang':'1.12.0'}:
            raise ValueError(f'Unaudited Warp version: {versions}')
        self.model=model
        self.step_count=0
        self.raw_time=0.
        self.maximum_clock_error=0.
        self.maximum_occupancy={'broadphase_pairs':0,'contacts':0,'constraints':0}
        self.stop=threading.Event()
        self.sampler=threading.Thread(target=sample_resources,
            args=(os.getpid(),directory,self.stop),daemon=True)
        self.sampler.start()
        os.environ['TMPDIR']='/tmp'
        # Full validation reuses version-pinned compiled kernels. Cold-cache
        # benchmarking remains a separate module with a fresh cache per run.
        wp.config.kernel_cache_dir=str(directory.parent/'warp_evaluation_cache')
        t=time.perf_counter()
        wp.init(); wp.set_device('cuda:0')
        self.device=wp.get_device('cuda:0')
        if not self.device.is_cuda:
            raise RuntimeError('CUDA physics required; no CPU fallback')
        # Pinned Warp step() implements no automatic numerical reset. Omit
        # only its unsupported flag during transfer; native model retains it.
        flags=int(model.opt.disableflags)
        try:
            model.opt.disableflags=flags & ~int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
            self.wm=mjwarp.put_model(model)
        finally:
            model.opt.disableflags=flags
        self.wd=mjwarp.put_data(model,data,nworld=1,nconmax=256,njmax=1024)
        wp.synchronize()
        self.setup_s=time.perf_counter()-t
        t=time.perf_counter()
        # Compile first, then discard the warmup state completely.
        mjwarp.step(self.wm,self.wd); mjwarp.forward(self.wm,self.wd)
        wp.synchronize()
        self.compile_s=time.perf_counter()-t
        self.wd=mjwarp.put_data(model,data,nworld=1,nconmax=256,njmax=1024)
        with wp.ScopedCapture() as capture:
            mjwarp.step(self.wm,self.wd)
        self.graph=capture.graph
        with wp.ScopedCapture() as capture:
            mjwarp.forward(self.wm,self.wd)
        self.forward_graph=capture.graph
        self.info={'versions':versions,'device':self.device.name,'tensor_device':str(self.device),
            'kernel_cache':wp.config.kernel_cache_dir,
            'model_transfer_and_device_startup_s':self.setup_s,
            'first_step_and_forward_compilation_s':self.compile_s,
            'autoreset_semantics':'Warp has no numerical autoreset; finite-state and incremental-clock checks every physics step',
            'force_source':'get_data_into actual Warp constraint/actuator forces; no CPU dynamics recomputation',
            'source_sha256':digest(__file__),
            'warp_io_source_sha256':digest(Path(mjwarp.__file__).parent/'_src/io.py'),
            'warp_forward_source_sha256':digest(Path(mjwarp.__file__).parent/'_src/forward.py')}

    def check_health(self):
        counts={'broadphase_pairs':int(self.wd.ncollision.numpy()[0]),
                'contacts':int(self.wd.nacon.numpy()[0]),'constraints':int(self.wd.nefc.numpy()[0])}
        capacities={'broadphase_pairs':self.wd.naconmax,'contacts':self.wd.naconmax,'constraints':self.wd.njmax}
        for k,v in counts.items():self.maximum_occupancy[k]=max(self.maximum_occupancy[k],v)
        arrays={k:getattr(self.wd,k).numpy() for k in ('qpos','qvel','qacc','actuator_force','xpos','xmat','subtree_com')}
        arrays['constraint_force']=self.wd.efc.force.numpy()[0,:max(0,counts['constraints'])]
        validate_physics_health(counts,capacities,arrays)

    def step(self, data):
        self.wd.ctrl.assign(np.asarray(data.ctrl[None,:],dtype=np.float32))
        self.wd.xfrc_applied.assign(np.asarray(data.xfrc_applied[None,:,:],dtype=np.float32))
        wp.capture_launch(self.graph)
        wp.synchronize()
        self.check_health()
        wp.capture_launch(self.forward_graph)
        wp.synchronize()
        self.check_health()
        mjwarp.get_data_into(data,self.model,self.wd)
        delta=data.time-self.raw_time
        validate_physics_health({}, {}, {},clock_delta=delta,dt=self.model.opt.timestep)
        self.raw_time=float(data.time)
        self.step_count+=1
        exact_time=self.step_count*self.model.opt.timestep
        self.maximum_clock_error=max(self.maximum_clock_error,abs(self.raw_time-exact_time))
        # Frame timestamps count fixed integration steps in float64. Do not
        # write this value into GPU physics; retain its float32 clock separately.
        data.time=exact_time

    def close(self):
        self.stop.set(); self.sampler.join(timeout=3)
        self.info.update(raw_gpu_clock_s=self.raw_time,
            maximum_float32_clock_error_s=self.maximum_clock_error,
            maximum_occupancy=self.maximum_occupancy,
            health_checks='broadphase/contact/constraint capacities and finite state/acceleration/forces/transforms/COM before and after post-step forward',
            physics_steps=self.step_count)
        utilization=[]; memory=[]
        for line in (self.directory/'resources.jsonl').read_text().splitlines():
            csv=json.loads(line).get('nvidia_smi',{}).get('csv','')
            if csv:
                fields=[v.strip() for v in csv.splitlines()[0].split(',')]
                utilization.append(float(fields[4])); memory.append(float(fields[6]))
        self.info.update(gpu_peak_utilization_percent=max(utilization,default=0),
                         gpu_peak_vram_mib=max(memory,default=0))
        return self.info

"""Bounded Landau Warp throughput/parity probe. This never constitutes gate proof."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import time

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import psutil
import warp as wp

from algorithms.urdf_learn_wasd_walk.mujoco_backend import (
    OUTPUT, audit_model, build_model, digest, initialize, write_json,
)
from algorithms.urdf_learn_wasd_walk.mujoco_g1_benchmark import sample_resources


def run(args):
    if not 1 <= args.worlds <= 4096 or not 1 <= args.blocks <= 100:
        raise ValueError('Bounded probe requires 1..4096 worlds and 1..100 blocks')
    if Path(args.name).name != args.name:
        raise ValueError('Name must be a single directory component')
    out = OUTPUT / 'warp_benchmarks' / args.name
    out.mkdir(parents=True, exist_ok=False)
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'command': shlex.join([sys.executable, '-m',
            'algorithms.urdf_learn_wasd_walk.mujoco_warp_benchmark', *sys.argv[1:]]),
        'cwd': str(Path.cwd()), 'args': vars(args), 'seed': 42,
        'milestone_pass': False, 'auxiliary_assistance_coefficient': 0.,
        'scope': 'Physics throughput and short deterministic parity; no policy or gate proof',
        'environment': {k: os.environ.get(k) for k in (
            'WARP_CACHE_PATH', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS')},
        'versions': {p: importlib.metadata.version(p) for p in (
            'mujoco', 'mujoco-warp', 'warp-lang', 'numpy')},
        'source_hashes': {p.name: digest(p) for p in (
            Path(__file__), Path(__file__).with_name('mujoco_backend.py'),
            Path(__file__).with_name('model_spec.py'),
            Path(__file__).with_name('mujoco_g1_benchmark.py'))},
    }
    (out / 'source.py').write_text(Path(__file__).read_text())
    start = time.perf_counter()
    process = psutil.Process()
    stop = threading.Event()
    sampler = threading.Thread(target=sample_resources,
        args=(os.getpid(), out, stop), daemon=True)
    sampler.start()

    def save(stage):
        result['stage'] = stage
        result['elapsed_wall_s'] = time.perf_counter() - start
        result['rss_bytes'] = process.memory_info().rss
        write_json(out / 'result.json', result)

    try:
        # The task worker supplies a private /tmp. NVRTC rejects the long
        # repository output path as TMPDIR, while its persistent cache is fine.
        os.environ['TMPDIR'] = '/tmp'
        result['nvrtc_temporary_directory'] = '/tmp (private worker namespace)'
        if any(result['versions'][p] != v for p,v in {
            'mujoco':'3.5.0', 'mujoco-warp':'3.5.0', 'warp-lang':'1.12.0'}.items()):
            raise ValueError('Reset mapping is audited only for the pinned MuJoCo/Warp 3.5.0 and Warp 1.12.0')
        save('initializing_device')
        t = time.perf_counter()
        wp.config.kernel_cache_dir = str(out / 'warp_cache')
        result['fresh_kernel_cache'] = str(out / 'warp_cache')
        wp.init()
        device = wp.get_device(args.device)
        result['device'] = {'alias': str(device), 'name': device.name,
                            'is_cuda': device.is_cuda}
        result['device_startup_s'] = time.perf_counter() - t
        if args.device.startswith('cuda') and not device.is_cuda:
            raise RuntimeError('Requested CUDA device was not obtained')
        with wp.ScopedDevice(device):
            model, spec, xml = build_model(noslip_iterations=20)
            result['audit'] = audit_model(model, spec)
            result['validated_cpu_xml_sha256'] = hashlib.sha256(xml.encode()).hexdigest()
            # Warp 3.5 step() has no checkPos/checkVel/checkAcc or autoreset
            # implementation. Keep native MuJoCo autoreset disabled, omit only
            # the unimplemented flag on transfer, and check time/state below.
            result['reset_semantics'] = {
                'native_autoreset': 'disabled',
                'warp_autoreset': 'not implemented in pinned step()',
                'transferred_flag_omission': 'mjDSBL_AUTORESET',
                'warp_forward_source_sha256': digest(Path(mjwarp.__file__).parent / '_src/forward.py'),
                'checks': 'finite state and exact monotonically increasing simulation time',
            }

            def transfer_model(m):
                flags = int(m.opt.disableflags)
                try:
                    m.opt.disableflags = flags & ~int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
                    return mjwarp.put_model(m)
                finally:
                    m.opt.disableflags = flags

            try:
                exact_model = transfer_model(model)
                result['validated_cpu_config_supported'] = True
                del exact_model
            except (NotImplementedError, ValueError) as exc:
                result['validated_cpu_config_supported'] = False
                result['validated_cpu_config_error'] = f'{type(exc).__name__}: {exc}'
            save('compatibility_checked')
            if args.noslip_iterations == 0:
                model, spec, xml = build_model(noslip_iterations=0)
            elif not result['validated_cpu_config_supported']:
                save('unsupported_exact_configuration')
                return
            result['physics_change_from_validated_cpu'] = {
                'noslip_iterations': [20, args.noslip_iterations],
                'standing_must_be_revalidated': True,
            }
            result['benchmark_xml_sha256'] = hashlib.sha256(xml.encode()).hexdigest()
            result['build_timing'] = spec['backend_build_timing']
            (out / 'model.xml').write_text(xml)
            native = initialize(model, spec, pose='geometric')
            initial_qpos = native.qpos.copy()
            initial_ctrl = native.ctrl.copy()
            result['initial_state_sha256'] = hashlib.sha256(
                initial_qpos.tobytes() + initial_ctrl.tobytes()).hexdigest()
            t = time.perf_counter()
            wm = transfer_model(model)
            data = mjwarp.put_data(model, native, nworld=args.worlds,
                                   nconmax=256, njmax=1024)
            wp.synchronize_device(device)
            result['model_data_transfer_s'] = time.perf_counter() - t
            save('compiling_first_step')
            t = time.perf_counter()
            mjwarp.step(wm, data)
            wp.synchronize_device(device)
            result['first_step_compilation_and_execution_s'] = time.perf_counter() - t
            # Fresh state after compilation; graphs must bind these new arrays.
            data = mjwarp.put_data(model, native, nworld=args.worlds,
                                   nconmax=256, njmax=1024)
            t = time.perf_counter()
            graph = None
            if device.is_cuda:
                with wp.ScopedCapture() as capture:
                    for _ in range(10):
                        mjwarp.step(wm, data)
                graph = capture.graph
            wp.synchronize_device(device)
            result['ten_step_graph_capture_s'] = time.perf_counter() - t
            save('steady_state')
            rows, gpu_states, cpu_states = [], [], []
            for block in range(args.blocks):
                t = time.perf_counter()
                cpu_start = process.cpu_times()
                if graph is not None:
                    wp.capture_launch(graph)
                else:
                    for _ in range(10):
                        mjwarp.step(wm, data)
                wp.synchronize_device(device)
                elapsed = time.perf_counter() - t
                cpu_end = process.cpu_times()
                # Transfers and parity validation intentionally excluded above.
                positions = data.qpos.numpy()
                velocities = data.qvel.numpy()
                times = data.time.numpy()
                cpu_t = time.perf_counter()
                for _ in range(10):
                    mujoco.mj_step(model, native)
                cpu_s = time.perf_counter() - cpu_t
                max_nefc = int(data.nefc.numpy().max())
                total_contacts = int(data.nacon.numpy()[0])
                row = {
                    'block': block, 'simulation_time_s': (block + 1) * .02,
                    'warp_wall_s': elapsed, 'native_single_world_wall_s': cpu_s,
                    'physics_transitions_per_s': args.worlds * 10 / elapsed,
                    'control_transitions_per_s': args.worlds / elapsed,
                    'cpu_percent_one_core_100': 100 * (
                        cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system) / elapsed,
                    'max_cpu_warp_qpos_difference': float(np.max(np.abs(positions[0] - native.qpos))),
                    'max_cpu_warp_qvel_difference': float(np.max(np.abs(velocities[0] - native.qvel))),
                    'cpu_warp_root_position_error_m': float(np.linalg.norm(positions[0,:3] - native.qpos[:3])),
                    'max_cpu_warp_joint_position_error_rad': float(np.max(np.abs(positions[0,7:] - native.qpos[7:]))),
                    'max_world_qpos_difference': float(np.max(np.abs(positions - positions[0]))),
                    'maximum_constraints_per_world': max_nefc,
                    'total_contacts': total_contacts,
                    'allocated_total_contacts': data.naconmax,
                    'all_states_finite': bool(np.isfinite(positions).all() and np.isfinite(velocities).all()),
                    'time_matches_no_reset': bool(np.allclose(times, (block + 1) * .02, atol=2e-5)),
                }
                rows.append(row)
                gpu_states.append(positions[0].copy())
                cpu_states.append(native.qpos.copy())
                if (not row['all_states_finite'] or not row['time_matches_no_reset']
                        or max_nefc >= 1024 or total_contacts >= data.naconmax):
                    raise RuntimeError(f'Invalid state or constraint capacity: {row}')
            result['samples'] = rows
            warm = rows[1:] or rows
            wall = sum(r['warp_wall_s'] for r in warm)
            result['steady_physics_transitions_per_s'] = len(warm) * args.worlds * 10 / wall
            result['steady_control_transitions_per_s'] = len(warm) * args.worlds / wall
            result['parity_max_qpos_difference'] = max(r['max_cpu_warp_qpos_difference'] for r in rows)
            result['parity_max_qvel_difference'] = max(r['max_cpu_warp_qvel_difference'] for r in rows)
            result['simulated_seconds_per_world'] = args.blocks * .02
            np.savez_compressed(out / 'parity_states.npz', cpu=cpu_states,
                                warp=gpu_states, initial=initial_qpos, ctrl=initial_ctrl)
            result['state_artifact_sha256'] = digest(out / 'parity_states.npz')
            save('completed')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        save('failed')
        raise
    finally:
        stop.set()
        sampler.join(timeout=3)
        if (out / 'resources.jsonl').exists():
            result['resources_sha256'] = digest(out / 'resources.jsonl')
            write_json(out / 'result.json', result)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--worlds', type=int, default=64)
    parser.add_argument('--blocks', type=int, default=50)
    parser.add_argument('--noslip-iterations', type=int, choices=(0, 20), default=20)
    run(parser.parse_args())

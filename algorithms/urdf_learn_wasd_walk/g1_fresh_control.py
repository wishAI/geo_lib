"""Train and continue our G1 lineage using the pinned official PPO recipe."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import threading
import time


def completed_swings(contacts, foot_positions, step_dt):
    """Count airborne bouts ending in contact, excluding contact chatter."""
    counts = [0, 0]
    for side in range(2):
        start = None
        baseline = peak = 0.
        for index in range(1, len(contacts)):
            if contacts[index - 1][side] and not contacts[index][side]:
                start = index
                baseline = foot_positions[index - 1][side][2]
                peak = foot_positions[index][side][2]
            if start is not None:
                peak = max(peak, foot_positions[index][side][2])
                if contacts[index][side]:
                    if (index - start) * step_dt >= .06 and peak - baseline >= .015:
                        counts[side] += 1
                    start = None
    return counts


def native_evaluate(args):
    """Independent CPU deployment-scene check; never executes bundled weights."""
    import copy
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from algorithms.urdf_learn_wasd_walk import mujoco_g1_control as control
    from algorithms.urdf_learn_wasd_walk import mujoco_g1_benchmark as benchmark
    torch.set_num_threads(1)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint.relative_to(benchmark.BACKEND.resolve())
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    actor = MLPModel(TensorDict({'actor': torch.zeros(1, 98)}, batch_size=[1]),
                     {'actor': ['actor']}, 'actor', 29, hidden_dims=[512, 256, 128],
                     activation='elu', obs_normalization=True,
                     distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar'})
    actor.load_state_dict(blob['actor_state_dict'], strict=True)
    actor.eval()

    class TorchSession:
        def get_inputs(self):
            return [SimpleNamespace(name='obs', shape=[1, 98])]

        def get_outputs(self):
            return [SimpleNamespace(name='action', shape=[1, 29])]

        def run(self, outputs, feed):
            with torch.inference_mode():
                obs = torch.as_tensor(feed['obs'], dtype=torch.float32)
                return [actor(TensorDict({'actor': obs}, batch_size=[len(obs)])).numpy()]

    original_load = control.load_control

    def load_our_control(repo):
        # Replace construction itself, so no bundled pretrained weights are loaded.
        with patch.object(control.ort, 'InferenceSession', return_value=TorchSession()):
            values = list(original_load(repo))
        provenance = copy.deepcopy(values[-1])
        provenance['files_sha256'] = {k: v for k, v in provenance['files_sha256'].items() if not k.endswith('.onnx')}
        provenance.update(baseline_kind='our_fresh_policy_in_independent_native_cpu_deployment_scene',
                          checkpoint=str(checkpoint), checkpoint_sha256=benchmark.sha(checkpoint),
                          pretrained_weights_loaded=False, controller_source_sha256=benchmark.sha(__file__),
                          scope='Deployment configuration is a separate transfer diagnostic; compare GPU evaluator too')
        values[-1] = provenance
        return tuple(values)

    control.load_control = load_our_control
    directory = benchmark.BACKEND / args.name
    outcome = control.run(SimpleNamespace(repo=benchmark.REPO, output=directory, seed=args.seed,
                                         duration=args.seconds, forward=.5, strafe=0., yaw=0.,
                                         video=args.video, audit_only=False))
    (directory / 'evaluator_source.py').write_text(Path(__file__).read_text())
    result = json.loads((directory / 'result.json').read_text())
    trace = np.load(directory / 'trajectory.npz')['physics_metrics']
    positions = np.zeros((len(trace), 2, 3))
    positions[:, :, 2] = trace[:, 5:7]
    swings = completed_swings(trace[:, 1:3].astype(bool), positions, .002)
    # Use clearance above each preceding stance, not the unsettled spawn height.
    passed = (result['failure'] is None and result['duration_s'] >= args.seconds - 1e-6
        and result['displacement_m'][0] >= .3 * args.seconds
        and abs(result['displacement_m'][1]) <= .1 * args.seconds
        and min(swings) >= 3 and result['max_tilt_rad'] < .6
        and max(result['foot_contact_slip_rms_m_s']) <= .2
        and result['max_both_feet_flight_s'] <= .12)
    benchmark.save(directory / 'independent_acceptance.json', {
        'status': 'passed' if passed else 'failed', 'checkpoint_sha256': benchmark.sha(checkpoint),
        'checkpoint': str(checkpoint), 'completed_swings_with_15mm_clearance': swings,
        'result_sha256': benchmark.sha(directory / 'result.json'),
        'minimum_swing_air_s': .06, 'minimum_swing_clearance_m': .015,
        'clearance_reference': 'last grounded foot position before each swing',
        'pretrained_weights_loaded': False, 'source_sha256': benchmark.sha(__file__)})
    if (directory / 'proof.mp4').is_file():
        benchmark.save(directory / 'proof_metadata.json', {
            'kind': 'same_rollout_native_cpu_deployment_scene',
            'video_sha256': benchmark.sha(directory / 'proof.mp4'),
            'trajectory_sha256': benchmark.sha(directory / 'trajectory.npz'),
            'result_sha256': benchmark.sha(directory / 'result.json')})
    # Execution completion and locomotion acceptance are distinct in result.json.
    return outcome


def training_worker(config_path):
    """Record optimizer diagnostics; any curriculum changes are explicit."""
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_g1_benchmark as benchmark
    from rsl_rl.algorithms import PPO
    config = json.loads(Path(config_path).read_text())
    directory = Path(config['directory'])
    original_update = PPO.update

    def update(algorithm):
        values = []
        get_kl = algorithm.actor.get_kl_divergence
        lr_before = algorithm.learning_rate

        def measured_kl(*args, **kwargs):
            kl = get_kl(*args, **kwargs)
            values.append(kl.mean().detach())
            return kl

        algorithm.actor.get_kl_divergence = measured_kl
        try:
            result = original_update(algorithm)
        finally:
            algorithm.actor.get_kl_divergence = get_kl
        kls = torch.stack(values).cpu().tolist() if values else []
        benchmark.append(directory / 'optimizer_diagnostics.jsonl', {
            'lr_before': lr_before, 'lr_after': algorithm.learning_rate,
            'first_pre_gradient_kl': kls[0] if kls else None,
            'mean_minibatch_kl': sum(kls) / len(kls) if kls else None,
            'max_minibatch_kl': max(kls) if kls else None})
        return result

    PPO.update = update
    if config.get('recipe') == 'precise_yaw':
        import mjlab.tasks.registry as registry
        original_env_cfg = registry.load_env_cfg

        def env_cfg(task, *args, **kwargs):
            cfg = original_env_cfg(task, *args, **kwargs)
            if task == 'Unitree-G1-Flat':
                cfg.rewards['track_angular_velocity'].params['std'] = .2
            return cfg
        registry.load_env_cfg = env_cfg
    benchmark.worker(config_path)


def train(args):
    from algorithms.urdf_learn_wasd_walk import mujoco_g1_benchmark as benchmark
    directory = (benchmark.BACKEND / args.name).resolve()
    directory.relative_to(benchmark.BACKEND.resolve())
    if subprocess.check_output(['git', '-C', str(benchmark.REPO), 'rev-parse', 'HEAD'], text=True).strip() != benchmark.REVISION:
        raise ValueError('Official source revision changed')
    subprocess.run(['git', '-C', str(benchmark.REPO), 'diff', '--exit-code', 'HEAD', '--', 'scripts/train.py', 'src/tasks/velocity', 'src/assets/robots/unitree_g1'], check=True, stdout=subprocess.DEVNULL)
    directory.mkdir(parents=True, exist_ok=False)
    overrides = {'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                 'WANDB_MODE': 'disabled', 'PYTHONUNBUFFERED': '1',
                 'PYTHONPATH': os.pathsep.join((str(benchmark.ROOT), str(benchmark.REPO))), 'TMPDIR': '/tmp'}
    for variable, folder in {'WARP_CACHE_PATH': 'warp_cache', 'XDG_CACHE_HOME': 'cache', 'TORCH_HOME': 'torch_cache',
                             'CUDA_CACHE_PATH': 'cuda_cache', 'MPLCONFIGDIR': 'mpl', 'TORCH_EXTENSIONS_DIR': 'torch_extensions',
                             'TRITON_CACHE_DIR': 'triton', 'XDG_CONFIG_HOME': 'config'}.items():
        (directory / folder).mkdir()
        overrides[variable] = str(directory / folder)
    if args.kernel_cache:
        cache = Path(args.kernel_cache).resolve()
        cache.relative_to(benchmark.BACKEND.resolve())
        shutil.copytree(cache, directory / 'warp_cache', dirs_exist_ok=True)
    argv = [str(benchmark.REPO / 'scripts/train.py'), 'Unitree-G1-Flat', '--env.scene.num-envs', str(args.num_envs),
            '--agent.seed', '42', '--agent.max-iterations', str(args.iterations), '--agent.save-interval', '250',
            '--agent.logger', 'tensorboard', '--agent.run-name', args.name]
    parent = None
    if args.resume_checkpoint:
        checkpoint = Path(args.resume_checkpoint).resolve()
        checkpoint.relative_to(benchmark.BACKEND.resolve())
        parent_dir = next((p for p in checkpoint.parents if (p / 'result.json').is_file()
                           and (p / 'benchmark_config.json').is_file()), None)
        if parent_dir is None:
            raise ValueError('Resume checkpoint requires our recorded training provenance')
        parent_result = json.loads((parent_dir / 'result.json').read_text())
        if parent_result.get('pretrained_weights_loaded') is not False or parent_result.get('returncode') != 0:
            raise ValueError('Resume must continue a completed run from our own fresh lineage')
        sha = benchmark.sha(checkpoint)
        if parent_result.get('checkpoint_sha256', {}).get(str(checkpoint.relative_to(parent_dir))) != sha:
            raise ValueError('Resume checkpoint hash differs from recorded training result')
        resume_dir = directory / 'logs/rsl_rl/g1_velocity/resume_parent'
        resume_dir.mkdir(parents=True)
        shutil.copy2(checkpoint, resume_dir / 'checkpoint.pt')
        argv += ['--agent.resume', 'True', '--agent.load-run', '^resume_parent$',
                 '--agent.load-checkpoint', '^checkpoint[.]pt$']
        parent = {'checkpoint': str(checkpoint), 'sha256': sha, 'training_directory': str(parent_dir),
                  'restore': 'Official runner restores actor, critic, optimizer, normalizers and curriculum step counter'}
    config = {'directory': str(directory), 'num_envs': args.num_envs, 'iterations': args.iterations, 'seed': 42,
              'env_overrides': overrides, 'official_argv': argv, 'timeout_s': args.timeout_s, 'cold_cache': not bool(args.kernel_cache), 'kernel_cache_source': args.kernel_cache,
              'render': False, 'resume': bool(parent), 'parent': parent, 'pretrained_checkpoint': None,
              'recipe': args.recipe, 'optimizer_diagnostics': True}
    if args.recipe == 'precise_yaw':
        config['recipe_overrides'] = {'track_angular_velocity_std': .2,
            'other_rewards_commands_physics_randomization': 'unchanged'}
    benchmark.save(directory / 'benchmark_config.json', config)
    (directory / 'control_source.py').write_text(Path(__file__).read_text())
    command = [sys.executable, str(Path(__file__).resolve()), '--training-worker', str(directory / 'benchmark_config.json')]
    began = time.perf_counter()
    with (directory / 'train.log').open('w') as log:
        child = subprocess.Popen(command, cwd=directory, env={**os.environ, **overrides}, stdout=log, stderr=subprocess.STDOUT)
        stop = threading.Event()
        sampler = threading.Thread(target=benchmark.sample_resources, args=(child.pid, directory, stop), daemon=True)
        sampler.start()
        timed_out = False
        try:
            try:
                code = child.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                child.terminate()
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            stop.set()
            sampler.join(timeout=3)
    result = benchmark.summarize(directory, config, command, time.perf_counter() - began, code, timed_out)
    result.update(created_at=datetime.now(timezone.utc).isoformat(), training_from_scratch=not bool(parent),
                  fresh_lineage=True, parent=parent,
                  pretrained_weights_loaded=False, recipe=args.recipe,
                  source_revision=benchmark.REVISION, control_source_sha256=benchmark.sha(directory / 'control_source.py'))
    benchmark.save(directory / 'result.json', result)
    print(json.dumps({key: result[key] for key in ('completed_iterations', 'wall_s', 'returncode', 'training_from_scratch', 'control_transitions_per_s_total_steady')}))
    if code:
        raise RuntimeError(f'G1 training exited {code}; evidence retained')


def evaluate(args):
    from dataclasses import asdict
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from algorithms.urdf_learn_wasd_walk import mujoco_g1_benchmark as benchmark
    sys.path.insert(0, str(benchmark.REPO))
    import mjlab.tasks
    import src.tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint.relative_to(benchmark.BACKEND.resolve())
    directory = (benchmark.BACKEND / args.name).resolve()
    directory.relative_to(benchmark.BACKEND.resolve())
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'evaluator_source.py').write_text(Path(__file__).read_text())
    import warp as wp
    if args.kernel_cache:
        cache = Path(args.kernel_cache).resolve()
        cache.relative_to(benchmark.BACKEND.resolve())
        shutil.copytree(cache, directory / 'warp_cache')
    wp.config.kernel_cache_dir = str(directory / 'warp_cache')
    cfg = load_env_cfg('Unitree-G1-Flat', play=True)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.viewer.width, cfg.viewer.height = 640, 480
    # Nominal, fixed-command comparison: overrides are explicit in the evidence.
    disabled = [] if args.randomized else ['foot_friction', 'encoder_bias', 'base_com']
    for name in disabled:
        cfg.events.pop(name, None)
    for axis in ('x', 'y', 'yaw'):
        cfg.events['reset_base'].params['pose_range'][axis] = (0., 0.)
    command = cfg.commands['twist']
    command.heading_command = args.heading_hold
    command.ranges.heading = (0., 0.) if args.heading_hold else None
    command.ranges.lin_vel_x = (.5, .5)
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (-1., 1.) if args.heading_hold else (0., 0.)
    command.rel_standing_envs = 0.
    command.init_velocity_prob = 0.
    command.resampling_time_range = (1e9, 1e9)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode='rgb_array')
    agent_cfg = load_rl_cfg('Unitree-G1-Flat')
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls('Unitree-G1-Flat') or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device='cuda:0')
    runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location='cuda:0')
    policy = runner.get_inference_policy(device='cuda:0')
    obs, _ = wrapped.reset()
    robot = env.scene['robot']
    foot_ids, foot_names = robot.find_sites(('left_foot', 'right_foot'))
    initial = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    samples, states, frames = [], [], []
    done_count = 0
    began = time.perf_counter()
    fps = round(1 / env.step_dt)
    try:
        with imageio.get_writer(directory / 'proof.mp4', fps=fps, codec='libx264') as writer:
            for step in range(round(args.seconds / env.step_dt)):
                # Save pre-step states; if auto-reset fires, its reset state is never counted or rendered.
                pos = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
                velocity = robot.data.root_link_lin_vel_b[0].detach().cpu().numpy().copy()
                quat = robot.data.root_link_quat_w[0].detach().cpu().numpy().copy()
                tilt = float(np.arccos(np.clip(1 - 2 * (quat[1]**2 + quat[2]**2), -1, 1)))
                feet = robot.data.site_pos_w[0, foot_ids].detach().cpu().numpy().copy()
                contact = env.scene['feet_ground_contact'].data.found[0].detach().cpu().numpy().copy()
                actual_command = env.command_manager.get_command('twist')[0].detach().cpu().numpy().copy()
                heading = float(robot.data.heading_w[0])
                valid_command = np.allclose(actual_command[:2], [.5, 0], atol=1e-6)
                valid_command &= abs(actual_command[2]) <= 1. + 1e-6 if args.heading_hold else abs(actual_command[2]) <= 1e-6
                if not valid_command:
                    raise ValueError(f'Evaluation command changed: {actual_command}')
                states.append(env.sim.data.qpos[0].detach().cpu().numpy().copy())
                samples.append({'time_s': step * env.step_dt, 'root_position_m': pos.tolist(),
                                'body_velocity_mps': velocity.tolist(), 'tilt_rad': tilt,
                                'foot_positions_m': feet.tolist(), 'foot_contact': contact.tolist(),
                                'command': actual_command.tolist(), 'heading_rad': heading})
                frame = env.render()
                writer.append_data(frame)
                if step % fps == 0:
                    frames.append(frame.copy())
                with torch.inference_mode():
                    obs, _, dones, _ = wrapped.step(policy(obs))
                if bool(dones.any()):
                    done_count = 1
                    break
        duration = len(samples) * env.step_dt
        # Only use the post-step endpoint if it is not an automatic reset.
        final = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy() if not done_count else np.array(samples[-1]['root_position_m'])
        forward, lateral = float(final[0] - initial[0]), float(final[1] - initial[1])
        speed = np.array([r['body_velocity_mps'][0] for r in samples])
        tilt = max(r['tilt_rad'] for r in samples)
        contact = np.asarray([r['foot_contact'] for r in samples]).reshape(len(samples), 2, -1).any(axis=2)
        lifts = np.sum(contact[:-1] & ~contact[1:], axis=0).tolist()
        foot_positions = np.asarray([r['foot_positions_m'] for r in samples])
        swings = completed_swings(contact, foot_positions, env.step_dt)
        clearance = (foot_positions[:, :, 2].max(axis=0) - foot_positions[:, :, 2].min(axis=0)).tolist()
        loaded = contact[1:] & contact[:-1]
        foot_speed = np.linalg.norm(np.diff(foot_positions, axis=0)[:, :, :2] / env.step_dt, axis=2)
        slip_rms = [float(np.sqrt(np.mean(foot_speed[:, side][loaded[:, side]]**2))) if loaded[:, side].any() else None for side in range(2)]
        both_air_fraction = float(np.mean(~contact.any(axis=1)))
        full = not done_count and duration >= args.seconds - 1e-6
        # This is a G1 nominal velocity-control diagnostic, never a Landau milestone.
        passed = full and forward >= .3 * args.seconds and abs(lateral) <= .1 * args.seconds and tilt < .6 and min(swings) >= 3 and min(clearance) >= .015 and all(value is not None and value <= .3 for value in slip_rms) and both_air_fraction <= .1
        np.savez_compressed(directory / 'trajectory.npz', qpos=np.array(states), time=np.arange(len(states))*env.step_dt)
        metrics = {'duration_s': duration, 'forward_m': forward, 'lateral_m': lateral,
                   'mean_forward_velocity_mps': float(speed.mean()), 'velocity_rmse_mps': float(np.sqrt(np.mean((speed - .5)**2))),
                   'max_tilt_rad': tilt, 'done_count': done_count, 'reset_count': done_count,
                   'foot_liftoffs': lifts, 'completed_swings': swings, 'foot_height_range_m': clearance, 'contact_slip_rms_mps': slip_rms, 'both_feet_air_fraction': both_air_fraction, 'body_height_min_m': min(r['root_position_m'][2] for r in samples)}
        result = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed' if passed else 'failed',
                  'scope': ('Randomized' if args.randomized else 'Nominal') + ' 0.5 m/s G1 diagnostic ' + ('with official heading-hold commands' if args.heading_hold else 'with fixed zero-yaw command') + '; not robustness certification or Landau evidence',
                  'checkpoint': str(checkpoint), 'checkpoint_sha256': benchmark.sha(checkpoint),
                  'pretrained_weights_loaded': False, 'metrics': metrics, 'samples': samples,
                  'evaluation_overrides': {'startup_randomization_disabled': disabled, 'initial_xy_yaw': 0,
                                          'command': [.5, 0, 'heading feedback' if args.heading_hold else 0], 'heading_hold': args.heading_hold,
                                          'heading_command_validation': 'fixed linear command and bounded official feedback; command/state update timing retained',
                                          'episode_timeout_s': 1e9, 'play_mode': True,
                                          'stop_on_first_done': True, 'seed': args.seed,
                                          'actual_command_checked_every_step': True},
                  'thresholds': {'min_average_forward_mps': .3, 'max_average_lateral_mps': .1, 'max_tilt_rad': .6, 'min_completed_swings_per_foot': 3, 'min_swing_air_time_s': .06, 'min_swing_clearance_m': .015, 'min_foot_height_range_m': .015, 'max_contact_slip_rms_mps': .3, 'max_both_air_fraction': .1},
                  'evaluator_source_sha256': benchmark.sha(directory / 'evaluator_source.py'), 'wall_s': time.perf_counter() - began}
        benchmark.save(directory / 'evaluation.json', result)
        benchmark.save(directory / 'proof_metadata.json', {'kind': 'same_rollout_render', 'frames': len(states), 'fps': fps,
                      'video_sha256': benchmark.sha(directory / 'proof.mp4'), 'trajectory_sha256': benchmark.sha(directory / 'trajectory.npz'),
                      'evaluation_sha256': benchmark.sha(directory / 'evaluation.json')})
        if frames:
            imageio.imwrite(directory / 'contact_sheet.png', np.concatenate([frames[0], frames[len(frames)//2], frames[-1]], axis=1))
        print(json.dumps({'status': result['status'], 'metrics': metrics, 'checkpoint': str(checkpoint)}))
    finally:
        wrapped.close()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--training-worker':
        training_worker(sys.argv[2])
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--mode', choices=['train', 'evaluate', 'native-evaluate'], default='train')
    p.add_argument('--checkpoint')
    p.add_argument('--resume-checkpoint', help='Exact checkpoint from our own completed fresh training lineage')
    p.add_argument('--recipe', choices=['official', 'precise_yaw'], default='official')
    p.add_argument('--kernel-cache')
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--seed', type=int, default=42, help='Evaluation seed; official training seed remains 42')
    p.add_argument('--randomized', action='store_true', help='Retain official startup randomization during evaluation')
    p.add_argument('--heading-hold', action='store_true', help='Use the official heading-to-yaw command feedback')
    p.add_argument('--video', action='store_true', help='Record independent native deployment playback')
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--timeout-s', type=int, default=900)
    args = p.parse_args()
    if not 1 <= args.num_envs <= 4096 or not 1 <= args.iterations <= 10001 or not 30 <= args.timeout_s <= 7000:
        raise ValueError('Bounded control experiment exceeds budget')
    if args.mode in ("evaluate", "native-evaluate"):
        if not args.checkpoint or not 0 < args.seconds <= 30:
            raise ValueError("Exact checkpoint and bounded duration required")
        (evaluate if args.mode == 'evaluate' else native_evaluate)(args)
    else:
        train(args)


if __name__ == '__main__':
    main()

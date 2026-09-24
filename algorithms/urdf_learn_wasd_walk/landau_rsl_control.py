"""Transfer the verified G1 PPO machinery to Landau's standing task only.

Runs inside the preserved task runtime. Geometry, gains, action bounds and the
named mass profile are frozen; this never promotes canonical milestones.
"""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import time


def configure_model(profile):
    import mujoco
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    from algorithms.urdf_learn_wasd_walk import mujoco_warp_batch as batch
    from algorithms.urdf_learn_wasd_walk.continuation import redistribute_mass
    original_build, original_audit = backend.build_model, backend.audit_model
    variant = {}

    def build(**kwargs):
        model, spec, xml = original_build(**kwargs)
        source = original_audit(model, spec)
        xml, record = redistribute_mass(xml, profile)
        variant.update(record, source_audit=source)
        return mujoco.MjModel.from_xml_string(xml), spec, xml

    def audit(model, spec):
        return {**variant['source_audit'], 'canonical_model': profile == 'original',
                'mass_variant': {k: v for k, v in variant.items() if k != 'source_audit'}}

    backend.build_model = batch.build_model = build
    backend.audit_model = batch.audit_model = audit
    return backend, batch, variant


def train(args):
    import mujoco
    import torch
    from tensordict import TensorDict
    from mjlab.rl import MjlabOnPolicyRunner, RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    backend, batch_module, variant = configure_model(args.profile)
    directory = (backend.OUTPUT / 'training' / args.name).resolve()
    directory.relative_to((backend.OUTPUT / 'training').resolve())
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    start = time.perf_counter()
    batch = batch_module.WarpBatch(args.num_envs, 42, 0., 'stand', 'single', .004, .4, .25)
    terminal_drift = torch.zeros(args.num_envs, device='cuda:0')
    original_reset = batch.reset

    def reset_with_terminal_metrics(mask):
        terminal_drift[mask] = (batch.pos[mask, batch.pelvis, :2] - batch.reference_xy).norm(dim=1)
        original_reset(mask)

    batch.reset = reset_with_terminal_metrics

    class StandingEnv:
        num_envs = args.num_envs
        num_actions = 17
        max_episode_length = 1500
        device = 'cuda:0'
        cfg = {'task': 'finite_horizon_30s_standing', 'reward_units': 'rate times 0.02 s'}
        common_step_counter = 0

        @property
        def unwrapped(self):
            return self

        @property
        def episode_length_buf(self):
            return batch.episode_steps

        def get_observations(self):
            displacement = torch.cat((batch.pos[:, batch.pelvis, :2] - batch.reference_xy,
                                      torch.zeros((args.num_envs, 1), device=self.device)), dim=1)
            local = torch.bmm(batch.rot[:, batch.base].transpose(1, 2), displacement[:, :, None]).squeeze(-1)
            remaining = 1. - batch.episode_steps.float() / self.max_episode_length
            obs = torch.cat((batch.observations(), local[:, :2] / .03, remaining[:, None]), dim=1)
            return TensorDict({'actor': obs, 'critic': obs}, batch_size=[self.num_envs])

        def step(self, actions):
            _, reward, done, timeout = batch.step(actions)
            drift = (batch.pos[:, batch.pelvis, :2] - batch.reference_xy).norm(dim=1)
            drift = torch.where(done.bool(), terminal_drift, drift)
            # Preserve terminal reward; reset positions must not alter its value.
            # Replace unbounded drift cost with a bounded proximity score: at
            # 3 cm, survival credit is exp(-1), and staying alive far away costs
            # less than deliberate termination. Other original penalties remain.
            penalty = 1. - torch.exp(-drift.square() / .03**2) - 8. * drift.square()
            reward = (reward - torch.where(done.bool() & ~timeout.bool(), 0., penalty)) * .02
            self.common_step_counter += 1
            return self.get_observations(), reward, done, {'log': {
                'Standing/drift_m': drift.mean(), 'Standing/done_fraction': done.mean(),
                'Standing/reward_per_step': reward.mean(),
                'Standing/completed_success_rate': sum(batch.completed) / max(1, len(batch.completed))}}

    cfg = RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(128, 128), activation='elu', obs_normalization=True,
            distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .1, 'std_type': 'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(128, 128), activation='elu', obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1., use_clipped_value_loss=True,
            clip_param=.2, entropy_coef=.01, num_learning_epochs=5, num_mini_batches=4,
            learning_rate=1e-4, schedule='adaptive', gamma=.99, lam=.95,
            desired_kl=.01, max_grad_norm=1.),
        experiment_name='landau_rsl_stand', save_interval=100, num_steps_per_env=24,
        max_iterations=args.iterations, logger='tensorboard', upload_model=False)
    config = asdict(cfg)
    env = StandingEnv()
    runner = MjlabOnPolicyRunner(env, copy.deepcopy(config), str(directory), device='cuda:0')
    last_layer = [m for m in runner.alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.orthogonal_(last_layer.weight, gain=.01)
    torch.nn.init.zeros_(last_layer.bias)
    metadata = {'created_at': datetime.now(timezone.utc).isoformat(), 'arguments': vars(args),
        'backend': 'mujoco_warp_cuda', 'mujoco_version': mujoco.__version__,
        'model_xml_sha256': hashlib.sha256(batch.xml.encode()).hexdigest(),
        'urdf_sha256': batch.spec['source']['urdf_sha256'],
        'mesh_tree_sha256': batch.spec['source']['mesh_tree_sha256'],
        'action_joints': batch.spec['action_joints'], 'nominal_q': batch.nominal_q.tolist(),
        'nominal_ctrl': batch.nominal_ctrl.tolist(), 'action_scale': .08,
        'observation_dim': 63, 'reference_xy': batch.reference_xy.tolist(),
        'mass_variant': variant, 'runner_config': config, 'canonical_milestone_pass': False,
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'mjlab', 'mujoco', 'mujoco-warp', 'rsl-rl-lib')},
        'source_sha256': {str(p): backend.digest(p) for p in (Path(__file__), Path(batch_module.__file__), Path(backend.__file__))},
        'transfer': ['official PPO and adaptive KL', 'saved observation normalization',
                     'reward rate times dt', 'observable local drift and remaining time'],
        'finite_horizon': '30 s true terminal with remaining time observed; no timeout bootstrap',
        'terminal_drift_recorded_before_reset': True,
        'initial_checkpoint': 'untrained interface baseline, never learned-standing evidence'}
    backend.write_json(directory / 'metadata.json', metadata)
    (directory / 'model.xml').write_text(batch.xml)
    (directory / 'control_source.py').write_text(Path(__file__).read_text())
    runner.save(str(directory / 'model_initial.pt'))
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    metadata.update(wall_s=time.perf_counter() - start, total_resets=batch.total_resets,
                    recent_30s_success_rate=sum(batch.completed) / max(1, len(batch.completed)),
                    checkpoints={p.name: backend.digest(p) for p in directory.glob('model_*.pt')})
    backend.write_json(directory / 'training.json', metadata)
    print(json.dumps({k: metadata[k] for k in ('wall_s', 'total_resets', 'recent_30s_success_rate')}))


def evaluate(args):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from algorithms.urdf_learn_wasd_walk import mujoco_policy as old_policy
    checkpoint = Path(args.checkpoint).resolve()
    meta = json.loads((checkpoint.parent / 'training.json').read_text())
    backend, _, _ = configure_model(meta['arguments']['profile'])
    checkpoint.relative_to((backend.OUTPUT / 'training').resolve())
    if backend.digest(checkpoint) != meta['checkpoints'][checkpoint.name]:
        raise ValueError('Checkpoint differs from completed training provenance')
    actor = MLPModel(TensorDict({'actor': torch.zeros(1, 63)}, batch_size=[1]),
        {'actor': ['actor']}, 'actor', 17, hidden_dims=[128, 128], activation='elu',
        obs_normalization=True,
        distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .1, 'std_type': 'scalar'})
    actor.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['actor_state_dict'], strict=True)
    actor.eval()
    original_observe = old_policy.observe

    def observe(model, data, nominal, joints, previous):
        base, pelvis = model.body('base_link').id, model.body('root_x').id
        displacement = np.r_[data.xpos[pelvis, :2] - np.array(meta['reference_xy']), 0.]
        local = data.xmat[base].reshape(3, 3).T @ displacement
        return np.r_[original_observe(model, data, nominal, joints, previous),
                     local[:2] / .03, 1. - data.time / 30.].astype(np.float32)

    def load_actor(path, model, spec):
        if meta['urdf_sha256'] != spec['source']['urdf_sha256'] or meta['mesh_tree_sha256'] != spec['source']['mesh_tree_sha256']:
            raise ValueError('Evaluation asset mismatch')

        def inference(obs):
            return actor(TensorDict({'actor': obs[None]}, batch_size=[1]))[0]

        return SimpleNamespace(actor=inference), meta

    old_policy.load_actor, old_policy.observe = load_actor, observe
    backend.run(SimpleNamespace(name=args.name, backend='mujoco_warp_cuda', seconds=30.,
        dt=.002, seed=42, pose='geometric', gain_scale=1., noslip_iterations=0,
        contact_timeconst=.004, assistance=0., checkpoint=str(checkpoint), forward=0., render=False))
    folder = backend.OUTPUT / args.name
    backend.write_json(folder / 'transfer.json', {'checkpoint': str(checkpoint),
        'checkpoint_sha256': backend.digest(checkpoint), 'initial_untrained': checkpoint.name == 'model_initial.pt',
        'profile': meta['arguments']['profile'], 'canonical_milestone_pass': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train')
    parser.add_argument('--profile', choices=['original', 'balanced_hands_v1'], default='balanced_hands_v1')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 512 or not 1 <= args.iterations <= 1000:
        raise ValueError('Standing transfer budget exceeded')
    if args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError('An exact checkpoint is required')
        evaluate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()

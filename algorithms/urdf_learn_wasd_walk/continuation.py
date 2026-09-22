"""Bounded continuation experiments in the preserved TK2 MuJoCo runtime.

This adapter is copied into the isolated runtime alongside its dependencies.
All outputs retain their own model identity; no canonical milestone is changed.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET


def redistribute_mass(xml, profile):
    """Move 25% of finger mass into pelvis/spine, scaling full inertia equally."""
    if profile not in ('original', 'balanced_hands_v1'):
        raise ValueError('Unknown mass profile')
    root = ET.fromstring(xml)
    bodies = {b.get('name'): b.find('inertial') for b in root.iter('body') if b.find('inertial') is not None}
    before = {name: float(i.get('mass')) for name, i in bodies.items()}
    after = before.copy()
    if profile != 'original':
        fingers = [name for name in bodies if name.startswith(('thumb', 'index', 'middle', 'ring', 'pinky'))]
        if len(fingers) != 38:
            raise ValueError('Unexpected finger topology')
        transferred = sum(before[name] * .25 for name in fingers)
        for name in fingers:
            after[name] *= .75
        recipients = {'root_x': .4, 'spine_01_x': .2, 'spine_02_x': .2, 'spine_03_x': .2}
        if not set(recipients) <= bodies.keys():
            raise ValueError('Unexpected torso topology')
        for name, fraction in recipients.items():
            after[name] += transferred * fraction
        for name, inertial in bodies.items():
            factor = after[name] / before[name]
            inertial.set('mass', str(after[name]))
            for key in ('fullinertia', 'diaginertia'):
                if key in inertial.attrib:
                    inertial.set(key, ' '.join(str(float(v) * factor) for v in inertial.get(key).split()))
    audit = {'profile': profile, 'total_mass_before_kg': sum(before.values()),
             'total_mass_after_kg': sum(after.values()), 'geometry_changed': False,
             'inertial_origins_changed': False, 'method': 'uniform per-link density scaling',
             'changes': {n: {'before_kg': before[n], 'after_kg': after[n]} for n in before if before[n] != after[n]},
             'milestone_pass': False}
    if abs(sum(after.values()) - sum(before.values())) > 1e-10:
        raise ValueError('Mass conservation failed')
    return ET.tostring(root, encoding='unicode') if profile != 'original' else xml, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', help='Existing result.json defining exact student and runtime settings')
    p.add_argument('--name')
    p.add_argument('--profile', choices=['original', 'balanced_hands_v1'], default='original')
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--stand', action='store_true')
    p.add_argument('--period', type=float)
    p.add_argument('--mode', choices=['student', 'passive', 'policy-eval', 'train-stand', 'render'], default='student')
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--checkpoint')
    p.add_argument('--render-directory', nargs='+')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    args = p.parse_args()
    if args.mode != 'render' and (not args.baseline or not args.name):
        p.error('--baseline and --name are required outside render mode')
    if args.mode == 'render' and not args.render_directory:
        p.error('--render-directory is required in render mode')
    if not 0 < args.seconds <= 120 or (args.period is not None and not 2 <= args.period <= 30):
        raise ValueError('Trial exceeds bounds')
    if not 1 <= args.iterations <= 1000 or not 0 < args.learning_rate <= .001:
        raise ValueError('Training budget or learning rate exceeds bounds')
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    if args.mode == 'render':
        import subprocess, sys
        for directory in args.render_directory or []:
            folder = Path(directory).resolve()
            folder.relative_to(backend.OUTPUT.resolve())
            subprocess.run([sys.executable, '-m', 'algorithms.urdf_learn_wasd_walk.mujoco_render', str(folder)], check=True, timeout=240)
            proof = json.loads((folder / 'proof_metadata.json').read_text())
            proof.update(dynamics_sha256=backend.digest(folder / 'dynamics.json'),
                         model_xml_sha256=backend.digest(folder / 'model.xml'))
            backend.write_json(folder / 'proof_metadata.json', proof)
        return
    from algorithms.urdf_learn_wasd_walk import mujoco_ragdoll_teacher as teacher
    import mujoco
    baseline_path = Path(args.baseline).resolve()
    baseline_path.relative_to(backend.OUTPUT.resolve())
    baseline = json.loads(baseline_path.read_text())
    original_build = backend.build_model
    original_audit = backend.audit_model
    variant = {}

    def build(**kwargs):
        nonlocal variant
        model, spec, xml = original_build(**kwargs)
        source_audit = original_audit(model, spec)
        xml, variant = redistribute_mass(xml, args.profile)
        if args.profile != 'original':
            model = mujoco.MjModel.from_xml_string(xml)
        variant['source_audit'] = source_audit
        return model, spec, xml

    def audit(model, spec):
        if args.profile == 'original':
            return original_audit(model, spec)
        # The exact canonical source was audited before the named variant was compiled.
        result = copy.deepcopy(variant['source_audit'])
        result['mass_variant'] = {k: v for k, v in variant.items() if k != 'source_audit'}
        result['canonical_model'] = False
        result['total_mass_kg'] = float(model.body_mass.sum())
        return result

    if args.mode == 'policy-eval' and not args.checkpoint:
        raise ValueError('Policy evaluation requires an exact checkpoint')
    backend.build_model = teacher.build_model = build
    backend.audit_model = teacher.audit_model = audit
    if args.mode == 'student':
        cfg = baseline['config'].copy()
        cfg.pop('motor_pose_sequence_sha256', None)
        cfg.update(name=args.name, seconds=args.seconds, coefficient=0., teacher_blend=0.,
                   motor_pose_sequence=None, teacher_tracking_integral=0.,
                   diagnostic_observation_replay=None, diagnostic_oracle_student=False,
                   teacher_rescue_time=None, motor_balance_gain=0.)
        if args.stand:
            cfg['stride'] = 0.
        if args.period:
            cfg['period'] = args.period
        teacher.run(SimpleNamespace(**cfg))
        out = backend.OUTPUT / 'ragdoll' / teacher.LINEAGE / args.name
    elif args.mode in ('passive', 'policy-eval'):
        cfg = dict(name=args.name, backend='mujoco_warp_cuda', seconds=args.seconds,
                   dt=.002, seed=42, pose='geometric', gain_scale=1., noslip_iterations=0,
                   contact_timeconst=.004, assistance=0., checkpoint=args.checkpoint if args.mode == 'policy-eval' else None, forward=0., render=False)
        backend.run(SimpleNamespace(**cfg))
        out = backend.OUTPUT / args.name
    else:
        from algorithms.urdf_learn_wasd_walk import mujoco_policy as policy
        from algorithms.urdf_learn_wasd_walk import mujoco_warp_batch as batch
        policy.build_model = batch.build_model = build
        batch.audit_model = audit
        cfg = dict(name=args.name, seed=42, backend='mujoco_warp_cuda', contact_timeconst=.004,
                   forward_exploration_multiplier=1., mean_activation='linear', num_envs=64,
                   iterations=args.iterations, budget_s=180., assistance=0., resume=None,
                   stage='stand', stand_checkpoint=None, forward_speed=.4,
                   forward_tracking_variance=.25, learning_rate=args.learning_rate, target_kl=.01,
                   gradient_clipping='separate', gait_reward='single')
        policy.train(SimpleNamespace(**cfg))
        out = backend.OUTPUT / 'training' / args.name
    print(json.dumps({'mass_variant': {k: v for k, v in variant.items() if k != 'source_audit'},
                      'mode': args.mode, 'name': args.name, 'canonical_milestones_modified': False}))
    # Training/dynamics keep their normal result files; attach the variant record to the run.
    if out.is_dir():
        (out / 'continuation_source.py').write_text(Path(__file__).read_text())
        backend.write_json(out / 'continuation.json', {'arguments': vars(args), 'source_sha256': backend.digest(__file__), 'baseline_result_sha256': backend.digest(baseline_path), 'canonical_milestones_modified': False})
        backend.write_json(out / 'mass_variant.json', {k: v for k, v in variant.items() if k != 'source_audit'})


if __name__ == '__main__':
    main()

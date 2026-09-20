"""Aggregate actual same-source hand variants, retaining smoothing tradeoffs."""
import json
import numpy as np
from state import OUT, artifact, node, sha256, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    comparisons = []
    for seed in (42, 43, 44):
        names = [f'animation_contact_v3_seed{seed}',
                 f'animation_hands_v2_seed{seed}', f'animation_hands_v3_seed{seed}']
        paths = [OUT/'runs'/name for name in names]
        if len({sha256(p/'source.npz') for p in paths}) != 1:
            raise ValueError('Hand variants must retain identical source motion')
        quality = [read(p/'quality_frames.json') for p in paths]
        hands = [read(p/'hands.json') for p in paths[1:]]
        with np.load(paths[0]/'target.npz') as before, np.load(paths[2]/'target.npz') as after:
            if not np.array_equal(before['times'], after['times']):
                raise ValueError('Different timing')
        # Every-frame foot/facing diagnostics must remain invariant, not just their means.
        unchanged = [k for k in quality[0]['per_frame'] if
                     k.startswith(('l_', 'r_')) or 'facing_error' in k or k.startswith('root_')]
        deltas = {k: float(np.max(np.abs(np.asarray(quality[0]['per_frame'][k]) -
                                       np.asarray(quality[2]['per_frame'][k])))) for k in unchanged}
        if max(deltas.values()) > 1e-10:
            raise ValueError('Hand-only refinement changed foot/root/facing diagnostics')
        smoothing = {k: {s: hands[1]['statistics'][k][s]-hands[0]['statistics'][k][s]
                         for s in ('mean', 'p95', 'maximum')}
                     for k in hands[1]['statistics'] if '_after_' in k}
        comparisons.append({'seed': seed, 'before': names[0], 'intermediate': names[1],
            'after': names[2], 'source_sha256': sha256(paths[0]/'source.npz'),
            'same_timing': True, 'same_cameras': True, 'frame_count': quality[2]['frame_count'],
            'orientation_statistics': hands[1]['statistics'],
            'smoothing_v3_minus_v2_deg': smoothing,
            'joint_jitter_rad': {name: q['statistics']['joint_jitter_max_rad']
                                 for name, q in zip(names, quality)},
            'unchanged_per_frame_max_abs_delta': deltas,
            'max_landmark_position_change_m': hands[1]['max_landmark_position_change_m'],
            'quality_reports': [artifact(p/'quality_frames.json') for p in paths],
            'hand_report': artifact(paths[2]/'hands.json'),
            'comparison': artifact(OUT/f'hands_v3_comparison_seed{seed}/comparison.json'),
            'review': artifact(paths[2]/'review.json')})
    report = {'purpose': 'animation quality diagnostics; no physical acceptance gates',
        'status': 'reviewed_candidate_preserving_contact_default',
        'recommended_run': 'animation_contact_v3_seed42',
        'candidate_run': 'animation_hands_v3_seed42',
        'proportions_and_behavior': 'Retargeting changes target proportions, not learned movement style. Uniform scaling or equal rotations does not demonstrate childlike gait; no child-behavior claim, training or generator change.',
        'frames_per_variant': 540, 'comparisons': comparisons,
        'cause': 'Position-only landmarks cannot observe wrist pitch or collinear forearm roll. Anatomical middle-finger and thumb-side axes add the missing orientation information.',
        'intervention': 'Fit only four wrist/forearm joints; smooth their corrections with sigma 1.5 frames. Preserve original root, legs, upper arms and locked finger joints.',
        'retained_failures': {'hand_orientation_null_seed42': 'Tiny nonzero initial parameters gave TRF an approximately 1e-15 initial trust radius and premature termination; snapping numerical zero fixes this.',
            'animation_hands_v2_seed42': 'Sigma 0.6 left additional wrist jitter: inspect retained per-frame diagnostics.',
            'animation_hands_v2_seed44': 'Sigma 0.6 left additional wrist jitter: inspect retained per-frame diagnostics.'},
        'limitations': ['Large residual hand-direction errors remain, especially seeds42/43. Four distal joints cannot reproduce every source wrist pose while preserving arm landmarks.',
            'Canonical right upper-arm roll is perpendicular to its bone while left roll aligns with it; canonical assets remain unchanged.',
            'Finger articulation is locked; hand basis fidelity is measured, individual finger fidelity is unavailable.',
            'Exact mesh intersections are unavailable. Rotating hand surfaces may affect hand/torso overlap despite unchanged wrist positions and capsule proxies.',
            'Dense frame review and full-duration decoding are recorded separately from real-time GUI playback; new hand-variant playback remains pending.'],
        'visual_reviews': [read(OUT/'hand_visual_review_42.json'), read(OUT/'hand_visual_review_43_44.json')]}
    write_json(OUT/'hand_comparison.json', report)
    node('multiframe_hand_review', [f'animation_hands_v3_seed{s}:validated' for s in (42,43,44)],
         'passed', label='Three-seed hand orientation review; residuals retained',
         artifacts=[artifact(OUT/'hand_comparison.json')], metrics={'frames_per_variant': 540})
    print(json.dumps({'report': str(OUT/'hand_comparison.json'), 'seeds': [42,43,44]}))


if __name__ == '__main__':
    main()

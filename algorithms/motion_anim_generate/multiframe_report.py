"""Compare retained-source contact refinements without turning diagnostics into gates."""
import json
import numpy as np
from state import OUT, artifact, node, sha256, write_json

PAIRS = [(42, 'animation_feet_facing_6s42'),
         (43, 'animation_multiframe_baseline43'), (44, 'animation_multiframe_baseline44')]


def main():
    comparisons = []
    for seed, before in PAIRS:
        after = f'animation_contact_v3_seed{seed}'
        paths = [OUT/'runs'/name for name in (before, after)]
        reports = [json.loads((p/'quality_frames.json').read_text()) for p in paths]
        if reports[0]['source_sha256'] != reports[1]['source_sha256']:
            raise ValueError('Different sources cannot isolate retarget refinement')
        with np.load(paths[0]/'target.npz') as a, np.load(paths[1]/'target.npz') as b:
            if not np.array_equal(a['times'], b['times']):
                raise ValueError('Different timing')
        metrics = {}
        for key in reports[0]['statistics']:
            a, b = [r['statistics'][key] for r in reports]
            metrics[key] = {'before': a, 'after': b, 'delta': {
                k: b[k]-a[k] for k in ('mean', 'p95', 'maximum')
                if a.get(k) is not None and b.get(k) is not None}}
        slide = []
        for report in reports:
            samples = []
            for side in ('l', 'r'):
                key = side+'_ankle_stance_slide_error_m_s'
                samples.extend(np.asarray(report['per_frame'][key])[report['measurement_masks'][key]])
            slide.append(float(np.mean(samples)))
        comparison = {'seed': seed, 'before': before, 'after': after,
            'source_sha256': reports[0]['source_sha256'], 'same_timing': True,
            'frames_per_version': reports[0]['frame_count'],
            'added_stance_slide_mean_m_s': {'before': slide[0], 'after': slide[1],
                'reduction_percent': 100*(1-slide[1]/slide[0])},
            'all_statistics': metrics,
            'comparison': artifact(OUT/f'contact_comparison_seed{seed}/comparison.json'),
            'reviews': [artifact(p/'review.json') for p in paths]}
        comparisons.append(comparison)
    report = {'purpose': 'animation quality, not control feasibility',
        'source_kind': 'three retained real unconditional Kimodo generations; no text semantics claimed',
        'numerically_measured_frames': sum(2*r['frames_per_version'] for r in comparisons),
        'intervention': 'Source-aware stance XY refinement with foot orientation prior; source displacement, swing targets and upper-body trajectory retained. Smooth only correction; preserve stance already active at clip boundaries.',
        'comparisons': comparisons,
        'review_method': 'All clean frames inspected in chronological sheets, plus 24 evenly spaced frames, numeric worst frames, contact/turn transitions and foot closeups. Full videos decode at original duration. Real-time GUI playback confirmation is separate and pending.',
        'tradeoffs': ['Reduced added stance drift does not mean zero absolute sliding; generated source contacts are estimates.',
            'Seed44 retains sole-orientation error during its turn and has a higher jitter percentile after refinement.',
            'Pose landmark changes and all orientation/swing statistics are retained above, including regressions.',
            'Upper-body pose/proportion mismatch and capsule intersection warnings remain; this leg-only refinement does not resolve them.'],
        'recommended_run': 'animation_contact_v3_seed42',
        'encoder_access': 'User authorized full encoder use; authenticated exact-revision probe returned HTTP403. Parent retries only after account approval.',
        'artifacts': [artifact(OUT/'runs/animation_contact_v3_seed42/clean_preview.mp4', 'video'),
                      artifact(OUT/'contact_comparison_seed42/preview.mp4', 'video')]}
    receipt=OUT/'parent_gui_review.json'
    if receipt.exists():
        report['parent_playback_review']=json.loads(receipt.read_text())
        report['parent_playback_receipt']={**artifact(receipt), 'sha256':sha256(receipt)}
        report['review_method']='All clean frames inspected in chronological sheets plus selected foot/source diagnostics. Parent real-time playback receipt recorded separately below.'
    write_json(OUT/'multiframe_comparison.json', report)
    node('multiframe_contact_review', [f'animation_contact_v3_seed{s}:validated' for s, _ in PAIRS],
         'passed', label='Three-seed animation review: stance drift reduced; tradeoffs retained',
         metrics={'measured_frames': report['numerically_measured_frames']},
         artifacts=[artifact(OUT/'multiframe_comparison.json')]+report['artifacts'])
    print(json.dumps([{'seed': r['seed'], **r['added_stance_slide_mean_m_s']} for r in comparisons]))


if __name__ == '__main__':
    main()

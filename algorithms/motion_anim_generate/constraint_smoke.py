"""Official empty-text pose conditioning and matched-noise constraint diagnostics."""
import numpy as np
from retarget import load_source
from state import OUT, sha256, write_json

DONOR = 'unconditional_cpu_6s_seed43'
FRAME = 30


def make_anchor(skeleton, device, run):
    import torch
    from kimodo.constraints import FullBodyConstraintSet
    path = OUT/'runs'/DONOR/'source.npz'
    source, names = load_source(path)
    indices = [dict((name, i) for i, (name, _) in enumerate(names))[name]
               for name in skeleton.bone_order_names]
    positions = source['posed_joints'][FRAME:FRAME+1, indices].copy()
    rotations = source['global_rot_mats'][FRAME:FRAME+1, indices].copy()
    translation = positions[0, skeleton.root_idx].copy()
    translation[1] = 0  # SOMA XZ ground plane, Y up. No Landau basis here.
    positions -= translation
    np.savez_compressed(run/'constraint_target.npz', positions=positions, rotations=rotations,
                        joint_names=np.asarray(skeleton.bone_order_names))
    anchor = FullBodyConstraintSet(skeleton,
        frame_indices=torch.tensor([FRAME], dtype=torch.long, device=device),
        global_joints_positions=torch.tensor(positions, dtype=torch.float32, device=device),
        global_joints_rots=torch.tensor(rotations, dtype=torch.float32, device=device),
        smooth_root_2d=None)
    write_json(run/'constraint_config.json', {
        'kind': 'official FullBodyConstraintSet; learned conditioning, no output clamping',
        'donor_run': DONOR, 'donor_sha256': sha256(path), 'donor_frame': FRAME,
        'output_anchor_frame': FRAME, 'soma_horizontal_translation_removed_m': translation.tolist(),
        'joint_names': skeleton.bone_order_names, 'cfg_type': 'separated', 'cfg_weight': [0., 2.],
        'text': '', 'text_guidance': 0., 'rotations_directly_constrained': False,
        'expected_diagnostic': {'position_rms_m_at_anchor_at_most': .05,
            'position_max_m_at_anchor_at_most': .10, 'rms_reduction_vs_matched_null_at_least': .5},
        'baseline': 'Same model, seed, initial heading, constraint input and separated CFG batch; only constraint guidance changes2 to0.',
        'not_claimed': 'No text semantics, training, physics or hard equality guarantee.'})
    return anchor


def measure_constraint(run, output, baseline, skeleton):
    import torch
    from kimodo.constraints import compute_global_heading
    from retarget import skeleton_names
    from scipy.spatial.transform import Rotation
    with np.load(run/'constraint_target.npz') as target:
        desired=target['positions'][0];rotations=target['rotations'][0]
    def metrics(sample):
        names=[name for name, _ in skeleton_names(sample['posed_joints'].shape[1])]
        indices=[names.index(name) for name in skeleton.bone_order_names]
        p=sample['posed_joints'][FRAME, indices]
        errors=np.linalg.norm(p-desired, axis=-1)
        root=p[skeleton.root_idx]-desired[skeleton.root_idx]
        h=compute_global_heading(torch.tensor(np.stack([desired,p]),device=skeleton.device),skeleton).cpu().numpy()
        rot=sample['global_rot_mats'][FRAME,indices]
        rotation_error=np.degrees(Rotation.from_matrix(rotations.transpose(0,2,1)@rot).magnitude())
        jump=np.linalg.norm(np.diff(sample['posed_joints'][FRAME-1:FRAME+2,indices],axis=0),axis=-1)
        return {'position_rms_m':float(np.sqrt(np.mean(errors**2))), 'position_max_m':float(errors.max()),
            'per_joint_position_error_m':errors.tolist(), 'root_horizontal_error_m':float(np.linalg.norm(root[[0,2]])),
            'root_height_error_m':float(abs(root[1])),
            'heading_error_deg':float(np.degrees(np.arccos(np.clip(np.dot(h[0],h[1]),-1,1)))),
            'rotation_error_deg_unconstrained':rotation_error.tolist(),
            'anchor_neighbor_max_joint_step_m':float(jump.max())}
    before,after=metrics(baseline),metrics(output)
    report={'frame':FRAME,'time_s':FRAME/30,'joint_names':skeleton.bone_order_names,
        'matched_null':before,'constraint_guided':after,
        'rms_reduction_fraction':1-after['position_rms_m']/before['position_rms_m'] if before['position_rms_m']>1e-8 else None,
        'purpose':'Constraint-following measurement; independent of animation-quality acceptance and text semantics.'}
    write_json(run/'constraint_metrics.json',report)
    return report

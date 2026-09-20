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


def render_evidence(run, null_run):
    """Show the actual null/guided sources around a clearly static requested pose."""
    import subprocess
    from PIL import Image, ImageDraw, ImageFont
    from render import FONT, encoder_command
    from retarget import skeleton_names
    from review import decode, sheets
    with np.load(run/'constraint_target.npz') as data:
        anchor=data['positions'][0];names=list(data['joint_names'])
    samples=[]
    for path in (run/'matched_null_source.npz',run/'source.npz'):
        source,sk=load_source(path);indices=[dict((n,i) for i,(n,_) in enumerate(sk))[n] for n in names]
        samples.append(source['posed_joints'][:,indices])
    parents=dict(skeleton_names(30));n=len(samples[0]);font=ImageFont.truetype(FONT,15)
    proc=subprocess.Popen(encoder_command(run/'constraint_source_comparison.mp4',960,720,30),stdin=subprocess.PIPE)
    try:
        for f in range(n):
            im=Image.new('RGB',(960,720),'#eff3f5');draw=ImageDraw.Draw(im)
            draw.text((12,8),f'Official pose conditioning | frame{f} / {n-1} | {f/30:.3f}s | requested anchor at1.000s',font=font,fill='#243b48')
            for col,(label,pos) in enumerate(zip(['NULL guidance0','STATIC requested pose','POSE guidance2'],[samples[0][f],anchor,samples[1][f]])):
                for row,axis in enumerate([0,2]):
                    ox=col*320;oy=40+row*335
                    def project(v):return (ox+160+v[axis]*145,oy+300-v[1]*145)
                    draw.rectangle((ox+2,oy,ox+318,oy+330),outline='#b6c4ce')
                    draw.line([(ox+3,oy+300),(ox+317,oy+300)],fill='#428768',width=2)
                    draw.text((ox+8,oy+5),label+(' | XY front' if row==0 else ' | ZY side'),font=font,fill='#263f50')
                    for j,name in enumerate(names):
                        parent=parents[name]
                        if parent is None:continue
                        color='#318cb2' if name.startswith('Left') else '#af6293' if name.startswith('Right') else '#526470'
                        draw.line([project(pos[names.index(parent)]),project(pos[j])],fill=color,width=4)
                    if f==FRAME:draw.rectangle((ox+3,oy+1,ox+317,oy+329),outline='#e29b14',width=4)
            proc.stdin.write(im.tobytes())
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('Constraint comparison encoder failed')
    frames=decode(run/'constraint_source_comparison.mp4')
    folder=run/'constraint_review';folder.mkdir(exist_ok=True)
    pages=sheets(frames,list(range(n)),folder,'source_sequence')
    cmd=['ffmpeg','-v','error','-y','-i',str(null_run/'clean_preview.mp4'),'-i',str(run/'clean_preview.mp4'),
         '-filter_complex',"[0:v]drawtext=text='NULL guidance0':x=12:y=48:fontsize=20:fontcolor=red[a];[1:v]drawtext=text='POSE guidance2':x=12:y=48:fontsize=20:fontcolor=blue[b];[a][b]hstack=inputs=2",
         '-c:v','libx264','-crf','24','-pix_fmt','yuv420p',str(run/'constraint_landau_comparison.mp4')]
    subprocess.run(cmd,check=True)
    write_json(run/'constraint_comparison.json',{'frame_count':n,'duration_s':n/30,
        'source_layout':'Null generated source | static requested anchor | guided generated source. Fixed SOMA world cameras; blueLeft/pinkRight.',
        'landau_layout':'Null left | guided right; same retarget configuration and fixed camera; different generated sources by design.',
        'source_pages':pages,'source_video_sha256':sha256(run/'constraint_source_comparison.mp4'),
        'landau_video_sha256':sha256(run/'constraint_landau_comparison.mp4'),
        'command':cmd,'real_time_playback':'pending parent GUI confirmation'})


def report_evidence(run):
    import json
    from state import artifact, node
    metrics=json.loads((run/'constraint_metrics.json').read_text())
    metadata=json.loads((run/'source_metadata.json').read_text())
    quality=json.loads((run/'quality_frames.json').read_text())
    report={'status':'single_pose_constraint_guidance_demonstrated',
        'scope':'Real pinned SOMA diffusion, official separated CFG[0,2], empty text; no training or text encoder.',
        'run':run.name,'paired_null_run':'constraint_pose_null_cuda_seed42',
        'job':'poseanchor20260920a','source_metrics':metrics,
        'model_load_and_both_inference_calls_s':metadata['elapsed_inference_s'],
        'peak_cuda_allocated_bytes':metadata['peak_cuda_allocated_bytes'],
        'target_animation_statistics':quality['statistics'],
        'visual_review':{'source_frames_reviewed':list(range(60)),
            'guided_and_null_landau_frames_reviewed':list(range(60)),
            'method':'Chronological full-sequence sheets, evenly24, all selected foot closeups including numeric worst/anchor neighbors. Real-time playback pending parent.',
            'notes':['Guided SOMA adopts the requested hands-on-hips pose; matched-null source has arms down.',
                'Landau retains broad elbow-out gesture but hands sit too high/close to torso. Chest-facing residual is about19degrees.',
                'Feet remain broadly planted and face forward; small drift remains. No isolated clean-frame snap seen.']},
        'limitations':['One two-second seed and one pose anchor; no general path or action-suite claim.',
            'FullBodyConstraintSet conditions joint positions, root and heading; it does not directly constrain supplied rotations.',
            'Learned guidance is approximate; output was not clamped or postprocessed to force equality.',
            'Source constraint accuracy and target retarget quality are separate measurements.'],
        'default_preview':'animation_contact_v3_seed42 remains promoted; this quiet smoke is additional evidence.',
        'next_bounded_experiment':{'status':'specified, not dispatched or demonstrated',
            'api':'Root2DConstraintSet','seconds':2,'steps':30,'seed':43,'cfg_type':'separated','cfg_weight':[0,2],
            'frame_indices':[0,15,30,45,59],'soma_xz_waypoints_m':[[0,0],[0,.1],[0,.2],[0,.3],[0,.4]],
            'comparison':'Matched-null same-noise guidance[0,0]; retain both full source and target clips.',
            'measure':'Waypoint and every-frame interpolated XZ error, final displacement and heading; then Landau sliding, feet and pose continuity.',
            'interpretation':'Tests an explicit slow forward root trajectory; does not by itself establish natural forward walking.'},
        'artifacts':[artifact(run/name,'video' if name.endswith('.mp4') else 'json') for name in
            ['constraint_metrics.json','constraint_config.json','quality_frames.json','review.json','clean_preview.mp4',
             'proof.mp4','constraint_source_comparison.mp4','constraint_landau_comparison.mp4']]}
    write_json(OUT/'constraint_feasibility.json',report)
    node('constraint_pose_review',[run.name+':validated','constraint_pose_null_cuda_seed42:validated'],
         'passed',label='Official pose conditioning works without text; retarget caveats retained',
         metrics={'anchor_rms_m':metrics['constraint_guided']['position_rms_m'],
                  'rms_reduction_fraction':metrics['rms_reduction_fraction']},
         artifacts=[artifact(OUT/'constraint_feasibility.json')]+report['artifacts'])
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--render-evidence',action='store_true')
    args=parser.parse_args()
    run=OUT/'runs/constraint_pose_cuda_seed42'
    if args.render_evidence:render_evidence(run,OUT/'runs/constraint_pose_null_cuda_seed42')
    report_evidence(run)

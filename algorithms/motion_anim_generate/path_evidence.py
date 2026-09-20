"""Review saved root-path inference; never invokes a generator or promotes a clip."""
import json
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from state import OUT, ROOT, artifact, sha256, write_json
from retarget import load_source
from constraint_smoke import PATH_FRAMES, PATH_XZ

RUN = 'constraint_root_cuda_seed43'
NULL = 'constraint_root_null_cuda_seed43'


def render_pair(run, null):
    from render import FONT, encoder_command
    from review import decode, sheets
    samples = [load_source(p/'source.npz')[0] for p in (null, run)]
    _, skeleton = load_source(run/'source.npz')
    names = [n for n, _ in skeleton]
    edges = [(i, names.index(parent)) for i, (_, parent) in enumerate(skeleton) if parent]
    font = ImageFont.truetype(FONT, 16)
    proc = subprocess.Popen(encoder_command(run/'path_source_comparison.mp4', 960, 720, 30), stdin=subprocess.PIPE)
    try:
        for f in range(60):
            im = Image.new('RGB', (960,720), '#eef2f4'); d = ImageDraw.Draw(im)
            d.text((12,8), f'Actual SOMA source | frame {f}/59 | {f/30:.3f}s | five sparse root waypoints', font=font, fill='#253c48')
            for col, (label, sample) in enumerate(zip(('NULL guidance0', 'PATH guidance2'), samples)):
                ox = col*480
                d.text((ox+12,38), label+' | fixed SOMA ZY side', font=font, fill='#253c48')
                def side(p): return ox+140+p[2]*155, 360-p[1]*155
                d.line([(ox+3,360),(ox+477,360)], fill='#40805a', width=2)
                for j,k in edges:
                    color = '#317fa5' if names[j].startswith('Left') else '#ad5c90' if names[j].startswith('Right') else '#506675'
                    d.line([side(sample['posed_joints'][f,k]),side(sample['posed_joints'][f,j])], fill=color, width=3)
                d.text((ox+12,390), 'XZ top: green requested | blue smooth root | red hips', font=font, fill='#253c48')
                def top(p): return ox+240+p[0]*350, 665-p[1]*350
                d.line([top(p) for p in PATH_XZ], fill='#328753', width=3)
                for frame,p in zip(PATH_FRAMES,PATH_XZ):
                    x,y=top(p);d.ellipse((x-4,y-4,x+4,y+4),fill='#328753')
                    d.text((x+8,y-8),str(frame),font=font,fill='#328753')
                for key,color in [('smooth_root_pos','#216bba'),('root_positions','#bf4f55')]:
                    points=sample[key][:f+1,[0,2]]
                    if len(points)>1:d.line([top(p) for p in points],fill=color,width=2)
                    x,y=top(points[-1]);d.ellipse((x-4,y-4,x+4,y+4),fill=color)
                d.rectangle((ox+2,32,ox+478,706),outline='#becbd1')
            proc.stdin.write(im.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait(): raise RuntimeError('Source comparison encoding failed')
    frames=decode(run/'path_source_comparison.mp4')
    folder=run/'path_review';folder.mkdir(exist_ok=True)
    pages=sheets(frames,list(range(60)),folder,'source_sequence')
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(null/'clean_preview.mp4'),'-i',str(run/'clean_preview.mp4'),
        '-filter_complex',"[0:v]drawtext=text='NULL guidance0':x=12:y=48:fontsize=20:fontcolor=red[a];[1:v]drawtext=text='PATH guidance2':x=12:y=48:fontsize=20:fontcolor=blue[b];[a][b]hstack=inputs=2",
        '-c:v','libx264','-crf','24','-pix_fmt','yuv420p',str(run/'path_landau_comparison.mp4')],check=True)
    paired=decode(run/'path_landau_comparison.mp4')
    assert len(paired)==len(frames)==60
    write_json(run/'path_comparison.json',{'source_pages':pages,'frame_count':60,'duration_s':2,
        'source_video_sha256':sha256(run/'path_source_comparison.mp4'),
        'landau_video_sha256':sha256(run/'path_landau_comparison.mp4'),
        'layout':'Null left, guided right; identical fixed cameras and original timing. Sources differ by learned constraint guidance; not a same-source retarget comparison.',
        'full_duration_decode':'passed; real-time GUI playback is separate'})


def main():
    from experiment import process
    from review import prepare
    run=OUT/'runs'/RUN;null=OUT/'runs'/NULL
    metadata=json.loads((run/'source_metadata.json').read_text())
    assert sha256(run/'matched_null_source.npz')==metadata['matched_null_source_sha256']
    for name in ('retarget.py','refine.py','landau.py','render.py'):
        if sha256(ROOT/name)!=metadata['implementation_sha256'][name]:
            raise ValueError('Guided/null implementation mismatch: '+name)
    if not null.exists():
        null.mkdir();shutil.copyfile(run/'matched_null_source.npz',null/'source.npz')
        meta={**metadata,'cfg_weight':[0.,0.],'source_kind':'local Kimodo matched-null constraint comparison',
            'source_sha256':sha256(null/'source.npz'),'source_node_id':RUN+':matched_null_generated',
            'paired_guided_run':RUN,'command':metadata['command'],
            'postprocess_command':'outputs/venv/bin/python path_evidence.py'}
        meta.pop('constraint_metrics',None)
        write_json(null/'source_metadata.json',meta)
        from state import node
        node(meta['source_node_id'],[RUN+':generated'],'passed',label='Saved matched-null generation; no new inference',artifacts=[artifact(null/'source.npz','file')])
        ret=json.loads((run/'retarget.json').read_text())
        process(null,'composite',meta['source_kind'],speed_bounded=ret['speed_bounded'],
            anchor_rigid=ret['anchor_rigid'],temporal_contact=True,
            animation_only=ret['temporal_contact']['animation_only'],foot_orientation=ret['orientation_constrained'],promote=False)
    for path in (run,null): prepare(path)
    for key in ('mapping','source_coordinate_matrix','anchor_rigid','speed_bounded','orientation_constrained','orientation_weights_m','active_joint_names'):
        values=[json.loads((p/'retarget.json').read_text())[key] for p in (run,null)]
        assert values[0]==values[1],key
    write_json(run/'retarget_comparison_config.json', {
        'same_settings_verified':True,
        'derived_root_trajectory_scales':{p.name:json.loads((p/'retarget.json').read_text())['root_trajectory_scale'] for p in (run,null)},
        'note':'Same retarget settings; trajectory scale is derived separately from each actual source. No re-generation or adjustment of guided target.'})
    render_pair(run,null)
    print(json.dumps({'guided':RUN,'null':NULL,'status':'rendered; visual review pending'}))


def publish_report():
    """Require actual visual-review receipts before publishing the bounded result."""
    from state import node
    run=OUT/'runs'/RUN;null=OUT/'runs'/NULL
    meta=json.loads((run/'source_metadata.json').read_text())
    implementation={name:sha256(ROOT/name) for name in ('retarget.py','refine.py','landau.py','render.py')}
    assert all(value==meta['implementation_sha256'][name] for name,value in implementation.items())
    from quality import summary
    target_translation={}
    for p in (run,null):
        ret=json.loads((p/'retarget.json').read_text());source,_=load_source(p/'source.npz')
        with np.load(p/'target.npz') as target:
            world=target['base'][:,:3,3];times=target['times']
        actual=world-world[0]
        desired=source['root_positions']@np.asarray(ret['source_coordinate_matrix']).T*ret['root_trajectory_scale']
        desired-=desired[0]
        error=np.linalg.norm((actual-desired)[:,:2],axis=-1)
        target_translation[p.name]={'world_xy_displacement_m':actual[-1,:2].tolist(),
            'mapped_source_hips_xy_displacement_m':desired[-1,:2].tolist(),
            'per_frame_xy_displacement_m':actual[:,:2].tolist(),
            'per_frame_added_horizontal_drift_m':error.tolist(),
            'added_horizontal_drift_m':summary(error,times,.02),
            'note':'Target world+Y forward; compare relative displacement to transported/scaled source hips, not unscaled SOMA smooth-root constraints.'}
    report={'status':'single_sparse_root_path_demonstrated_with_animation_limitations',
        'job':'rootpath20260920a','guided_run':RUN,'matched_null_run':NULL,
        'source_metrics':json.loads((run/'constraint_metrics.json').read_text()),
        'source_visual_review':json.loads((OUT/'path_source_visual_review.json').read_text()),
        'landau_visual_review':json.loads((OUT/'path_landau_visual_review.json').read_text()),
        'retarget_configuration':json.loads((run/'retarget_comparison_config.json').read_text()),
        'same_retarget_render_implementation_sha256':implementation,
        'target_statistics':{p.name:json.loads((p/'quality_frames.json').read_text())['statistics'] for p in (run,null)},
        'target_translation':target_translation,
        'parent_playback_receipt':json.loads((OUT/'parent_gui_review.json').read_text()),
        'model_load_plus_two_inference_calls_s':meta['elapsed_inference_s'],
        'peak_cuda_allocated_bytes':meta['peak_cuda_allocated_bytes'],
        'job_result':json.loads((OUT/'backend_gpu/results/rootpath20260920a.json').read_text()),
        'limits':['One two-second seed, five sparse smooth-root waypoints. This is not general arbitrary-path support, natural walking or childlike gait proof.',
            'Smooth root is the constrained model feature. Actual hips sway and their final displacement differs; both trajectories are retained.',
            'Guided and null are different generated motions with identical noise/retarget settings, not a same-source retarget-quality comparison.',
            'Generated/rendered is separate from quality: guided Landau retains twelve heuristic sliding frames and arm/chest pose differences.',
            'Foot contacts and capsule intersection checks are heuristic; exact mesh intersection is unavailable.',
            'All source and target frames were inspected in sheets and videos decoded at full duration. New path-video real-time GUI playback is not claimed.',
            'Text-conditioned idle/walk/turn/wave remains incomplete because official encoder account access is HTTP403; no retry or substitute encoder.'],
        'default_run':'animation_contact_v3_seed42',
        'next_step':'Bounded handoff complete; no new experiments. Text suite requires confirmed external encoder account approval.',
        'artifacts':[artifact(run/name,'video' if name.endswith('.mp4') else 'json') for name in
            ('path_source_comparison.mp4','path_landau_comparison.mp4','clean_preview.mp4','proof.mp4','constraint_metrics.json','quality_frames.json','review.json')]}
    write_json(OUT/'path_feasibility.json',report)
    node('root_path_review',[RUN+':validated',NULL+':validated'],'passed',
         label='Sparse root guidance reviewed; natural gait not established',
         artifacts=[artifact(OUT/'path_feasibility.json')]+report['artifacts'])
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish-reviewed',action='store_true',help='Aggregate existing visual-review receipts; no inference or rendering')
    args=parser.parse_args()
    if args.publish_reviewed: publish_report()
    else: main()

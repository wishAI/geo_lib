"""Aggregate generated animations separately from visual quality and text-access limits."""
import json
from collections import Counter
from state import OUT, write_json, sha256, progress, node, artifact
from validate import compare_semantics


def main():
    runs=[];parents=[];pending=[];failed=[]
    concerns={'floor_penetration','foot_sliding','self_intersection_proxy','quaternion_step','root_discontinuity','joint_continuity','joint_jitter','joint_limits'}
    for run in sorted((OUT/'runs').iterdir()):
        if not all((run/name).exists() for name in ('validation.json','video.json','retarget.json','source_metadata.json')):
            (failed if (run/'failure.json').exists() else pending).append(run.name);continue
        val=json.loads((run/'validation.json').read_text());meta=json.loads((run/'source_metadata.json').read_text())
        video=json.loads((run/'video.json').read_text());stream=video['ffprobe']['streams'][0]
        if int(stream['nb_frames'])!=val['frame_count']:raise ValueError('Incomplete video '+run.name)
        if abs(float(video['ffprobe']['format']['duration'])-val['duration_s'])>.04:raise ValueError('Wrong playback duration '+run.name)
        compare_semantics(run)
        paths=[run/name for name in ('source.npz','target.npz','source_metadata.json','retarget.json','source_validation.json',
            'validation.json','semantic_comparison.json','directions.json','proof.mp4','contact_sheet.png','video.json',
            'preview.mp4','clean_preview.mp4','clean_contact_sheet.png','clean_video.json','original_proof.mp4','original_video.json') if (run/name).exists()]
        runs.append({'id':run.name,'clip_status':'generated_and_rendered','source_kind':meta['source_kind'],
            'seed':meta.get('seed',meta.get('meta',{}).get('seed')),'duration_s':val['duration_s'],
            'model_revision':meta.get('pins',{}).get('model',{}).get('revision',meta.get('model_revision')),
            'inference_s':meta.get('elapsed_inference_s'),'legacy_kinematic_pass':val['kinematic_pass'],
            'legacy_pass_is_animation_acceptance':False,'metrics':val['metrics'],
            'quality_notes':sorted({v['check'] for v in val['violations'] if v['check'] in concerns}),
            'raw_violation_counts':dict(Counter(v['check'] for v in val['violations'])),
            'visual_review':video.get('visual_review','pending'),
            'artifacts':[{**artifact(p,'video' if p.suffix=='.mp4' else 'image' if p.suffix=='.png' else 'json' if p.suffix=='.json' else 'file'),
                          'size':p.stat().st_size,'sha256':sha256(p)} for p in paths]})
        parents.append(run.name+':validated')
    blockers=json.loads((OUT/'blockers.json').read_text())
    report={'status':'animation_pipeline_demonstrated_text_prompt_suite_awaiting_access','purpose':'animation generation and retargeting',
        'feasibility_conclusion':'Pinned Kimodo has generated real CPU and CUDA motion. Copied Landau animations render at full duration. Corrected root-frame and foot-orientation mapping substantially improve facing and foot pose. Visual quality concerns remain inspectable; actuator limits and arbitrary landmark RMSE are not rejection gates.',
        'recommended_run':'animation_feet_facing_6s42','preview':'algorithms/motion_anim_generate/outputs/preview.mp4',
        'requested_text_suite':{'status':'awaiting_authorized_encoder_or_exact_embeddings','actions':['idle','walk','turn','wave'],
            'smoke':{'seed':42,'seconds':2,'steps':30},'representative':{'seeds':[42,43,44],'seconds':6,'steps':100},
            'reason':'Gated Meta-Llama-3 encoder inaccessible; no official public prompt embeddings found. Unconditional samples are not claimed as requested actions.'},
        'physics':'outside animation scope; not an acceptance requirement','runs':runs,'failed_attempts':failed,'incomplete_runs':pending,
        'blockers':blockers['observations'],'cuda':json.loads((OUT/'cpu_cuda_comparison.json').read_text()),
        'diagnosis':['Old det=-1 axis swap plus hip-yaw placement reversed anatomical forward. Proper C and exact mounted-root mapping correct it.',
            'Ankle-to-toe displacement is not a sole normal. Explicit forward/up orientation residuals and unlocked shin twists improve feet.',
            'Original debug XZ depth sorting disagreed with its projection; updated before/after use identical correctly labeled fixed cameras.',
            'Position-only arm matching and contact heuristics remain approximate; foot sliding persists in the moving clip.',
            'Exact mesh self-intersection and source-to-target finger fidelity are unavailable. Capsule and contact diagnostics are heuristics.'],
        'next_step':'Inspect native clean preview and same-source comparison; resume conditional suite only with authorized pinned encoder or exact prompt embeddings.'}
    comparison=OUT/'facing_foot_comparison/comparison.json'
    if comparison.exists():report['facing_foot_comparison']=json.loads(comparison.read_text())
    write_json(OUT/'feasibility.json',report)
    node('animation_feasibility',parents,'passed',label='Real Landau animations rendered; visual notes available',
         artifacts=[artifact(OUT/'feasibility.json'),artifact(OUT/'preview.mp4','video')],metrics={'rendered_clips':len(runs)})
    progress('corrected_animation_reviewed_text_access_pending',active_process=None,active_run='animation_feet_facing_6s42',
        next_step=report['next_step'],blockers=blockers['observations'],gpu_dispatch=blockers['parent_dispatch'],
        measured_results=[{k:r[k] for k in ('id','clip_status','duration_s','inference_s','metrics','quality_notes')} for r in runs],
        artifacts=[artifact(OUT/'feasibility.json'),artifact(OUT/'preview.mp4','video'),artifact(OUT/'facing_foot_comparison/proof.mp4','video'),artifact(OUT/'evolution.json')])
    print(json.dumps({'rendered_clips':len(runs),'report':str(OUT/'feasibility.json')}))


if __name__=='__main__':main()

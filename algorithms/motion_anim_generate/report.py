"""Aggregate only existing evidence, including failures and explicit semantic blockers."""
import json
from collections import Counter
from state import ROOT, OUT, write_json, sha256, progress, node, artifact
from validate import compare_semantics


def main():
    runs=[];parents=[];pending=[]
    for run in sorted((OUT/'runs').iterdir()):
        if not all((run/name).exists() for name in ('validation.json','video.json','retarget.json','source_metadata.json')):
            pending.append(run.name);continue
        val=json.loads((run/'validation.json').read_text())
        meta=json.loads((run/'source_metadata.json').read_text())
        video=json.loads((run/'video.json').read_text())
        stream=video['ffprobe']['streams'][0]
        if int(stream['nb_frames'])!=val['frame_count']:raise ValueError('Incomplete video '+run.name)
        if abs(float(video['ffprobe']['format']['duration'])-val['duration_s'])>.04:raise ValueError('Wrong playback duration '+run.name)
        compare_semantics(run)
        paths=[run/name for name in ('source.npz','target.npz','source_metadata.json','retarget.json','source_validation.json',
                                     'validation.json','semantic_comparison.json','proof.mp4','contact_sheet.png','video.json')]
        runs.append({'id':run.name,'source_kind':meta['source_kind'],'seed':meta.get('seed',meta.get('meta',{}).get('seed')),
                     'model_revision':meta.get('pins',{}).get('model',{}).get('revision',meta.get('model_revision')),
                     'duration_s':val['duration_s'],'inference_s':meta.get('elapsed_inference_s'),
                     'kinematic_pass':val['kinematic_pass'],'metrics':val['metrics'],
                     'violation_counts':dict(Counter(v['check'] for v in val['violations'])),
                     'visual_review':video.get('visual_review','pending'),
                     'artifacts':[{**artifact(p,'video' if p.suffix=='.mp4' else 'image' if p.suffix=='.png' else 'json' if p.suffix=='.json' else 'file'),
                                   'size':p.stat().st_size,'sha256':sha256(p)} for p in paths]})
        parents.append(run.name+':validated')
    report={'status':'baseline_retargeting_failed_requested_text_suite_blocked',
            'feasibility_conclusion':'Pinned Kimodo inference and Landau retarget/render pipeline execute. This bounded position-IK baseline has not produced a kinematically valid target clip; it does not establish that improved retargeting is impossible.',
            'requested_text_suite':{'status':'blocked','actions':['idle','walk','turn','wave'],'smoke':{'seed':42,'seconds':2,'steps':30},
                                    'representative':{'seeds':[42,43,44],'seconds':6,'steps':100},
                                    'reason':'Official gated text encoder unavailable; no official public prompt embedding bundle found'},
            'dynamic_feasibility':'not tested','robot_control_safety':'not established; no hardware actuation',
            'runs':runs,'incomplete_runs':pending,'blockers':json.loads((OUT/'blockers.json').read_text()),
            'diagnosis':['Hard joint-speed bounds remove speed violations but leave acceleration, floor/contact and landmark-fit failures.',
                         'Landau proportions and available joint axes differ substantially from SOMA. Position-only IK leaves twist and upper-body orientation underconstrained.',
                         'Root transport is scaled but unconstrained by target support contacts; a single floor offset cannot prevent later penetration or sliding.',
                         'Capsule overlap is a heuristic; exact triangle collisions and physical tracking have not been measured.'],
            'bounded_alternatives_tested':['Unbounded per-frame IK vs URDF velocity-bounded IK on identical upstream source','2-second 30-step local unconditional smoke followed by three 6-second 100-step seeds'],
            'next_after_unblock':['Use authorized exact-prompt embeddings to run the requested smoke and seed suite.',
                                  'Improve rest-frame orientation matching and contact-aware trajectory IK with acceleration constraints; preserve current failed runs for comparison.']}
    write_json(OUT/'feasibility.json',report)
    node('feasibility',parents,'failed',label='Landau baseline fails; requested text suite blocked',
         artifacts=[artifact(OUT/'feasibility.json'),artifact(OUT/'blockers.json')],metrics={'valid_target_clips':sum(r['kinematic_pass'] for r in runs)})
    progress('blocked_requested_text_suite',active_process=None,active_run=None,
             next_step='Parent supplies authorized embeddings/encoder and permitted GPU dispatch; inspect feasibility.json and videos',
             measured_results=[{k:r[k] for k in ('id','duration_s','inference_s','kinematic_pass','metrics')} for r in runs],
             artifacts=[artifact(OUT/'feasibility.json'),artifact(OUT/'evolution.json'),artifact(OUT/'large_files.pending.json')],
             blockers=report['blockers']['observations'])
    print(json.dumps({'runs':len(runs),'valid_targets':sum(r['kinematic_pass'] for r in runs),'report':str(OUT/'feasibility.json')}))

if __name__=='__main__':main()

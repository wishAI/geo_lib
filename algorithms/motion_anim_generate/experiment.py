"""Reproducible file-based source -> Landau -> validation -> full-duration evidence."""
import argparse
import json
from pathlib import Path
import shutil
import time
from state import ROOT, OUT, progress, node, artifact, sha256, write_json
from retarget import solve
from validate import validate_file, validate_source, compare_semantics
from render import render


def process(run, action, source_kind, speed_bounded=False, anchor_rigid=True, temporal_contact=True,animation_only=True,foot_orientation=True):
    if animation_only:speed_bounded=False
    meta=json.loads((run/'source_metadata.json').read_text())
    write_json(run/'source_validation.json',validate_source(run/'source.npz'))
    source_id=meta.get('source_node_id',run.name+':generated');target_id=run.name+':retargeted';valid_id=run.name+':validated'
    if 'source_node_id' not in meta:
        node(source_id,['assets'],'passed',label=run.name+' · '+source_kind,
             provenance=meta,artifacts=[artifact(run/'source_metadata.json'),artifact(run/'source.npz','file')])
    progress('retargeting',active_run=run.name,next_step='Validate fitted Landau then render actual full source/target motion',
             config=meta,active_process='experiment.py CPU bounded IK')
    start=time.monotonic();ret=solve(run/'source.npz',run,speed_bounded=speed_bounded,anchor_rigid=anchor_rigid,foot_orientation=foot_orientation)
    if temporal_contact:
        from refine import temporal_contact_refine
        ret=temporal_contact_refine(run,animation_only=animation_only)
    node(target_id,[source_id],'passed',label=run.name+' · Landau retarget',
         metrics={'retarget_rmse_m':ret['retarget_rmse_m']},artifacts=[artifact(run/'retarget.json')])
    progress('validating',active_run=run.name,metrics=ret,next_step='Render complete front/side comparison')
    val=validate_file(run,action)
    if not val.get('animation_quality',{}).get('data_renderable'):
        raise ValueError('Malformed animation data; inspect validation.json')
    compare_semantics(run)
    from directions import diagnose
    diagnose(run)
    progress('rendering',active_run=run.name,metrics=val['metrics'],next_step='Review contact sheet and full-duration video')
    render(run,f'{source_kind} | {run.name}')
    val['clip_status']='generated_and_rendered'
    write_json(run/'validation.json',val)
    artifacts=[artifact(run/'validation.json'),artifact(run/'retarget.json'),artifact(run/'directions.json'),
               artifact(run/'proof.mp4','video'),artifact(run/'contact_sheet.png','image'),artifact(run/'video.json'),
               artifact(run/'clean_preview.mp4','video'),artifact(run/'clean_contact_sheet.png','image'),artifact(run/'clean_video.json')]
    node(valid_id,[target_id],'passed',label=run.name+' · animation rendered; see quality notes',
         metrics=val['metrics'],artifacts=artifacts,parameters={'source_kind':source_kind,'action':action})
    # Stable GUI preview aliases, with provenance in validation/report, original runs retained.
    for name in ['validation.json','proof.mp4','contact_sheet.png','preview.mp4','clean_preview.mp4','clean_contact_sheet.png']:
        shutil.copyfile(run/name,OUT/name)
    write_json(OUT/'latest.json',{'run':str(run),'source_kind':source_kind,'local_inference':source_kind.startswith('local Kimodo'),
                                'elapsed_s':time.monotonic()-start,'artifacts':artifacts})
    progress('evidence_ready_for_review',active_run=run.name,metrics=val['metrics'],artifacts=artifacts,
             active_process=None,next_step='Review evidence; gated text encoder still required for requested local prompt suite')
    return val


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--upstream-example',required=True,choices=['02_multi_text_prompt','05_root_path'])
    parser.add_argument('--run-id',required=True);parser.add_argument('--action',choices=['idle','walk','turn','wave','composite'],default='composite')
    parser.add_argument('--speed-bounded',action='store_true')
    args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run id')
    run=OUT/'runs'/args.run_id;run.mkdir(parents=True,exist_ok=False)
    folder=OUT/'vendor/kimodo/kimodo/assets/demo/examples/kimodo-soma-rp'/args.upstream_example
    shutil.copyfile(folder/'motion.npz',run/'source.npz')
    write_json(run/'source_metadata.json',{'source_kind':'upstream supplied example; no local inference',
        'upstream_example':args.upstream_example,'kimodo_revision':json.loads((ROOT/'provenance.json').read_text())['kimodo']['revision'],
        'model_revision':'not recorded in bundled example; do not attribute to selected v1.1 checkpoint',
        'meta':json.loads((folder/'meta.json').read_text()),'source_sha256':sha256(run/'source.npz'),
        'command':__import__('sys').argv})
    print(json.dumps(process(run,args.action,'upstream example (NO local inference)',args.speed_bounded)['metrics']))

if __name__=='__main__':main()

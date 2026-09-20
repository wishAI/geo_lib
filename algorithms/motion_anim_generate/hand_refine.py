"""Calibrated hand orientation for otherwise position-invisible forearm/wrist joints."""
import argparse
import json
import shutil
import subprocess
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from landau import Robot
from quality import summary
from retarget import load_source, skeleton_names, MAP
from state import ROOT, OUT, artifact, node, progress, sha256, write_json


def anatomical_basis(finger, thumb):
    u=np.asarray(finger,dtype=float);v=np.asarray(thumb,dtype=float)
    if u.shape!=(3,) or v.shape!=(3,) or not np.isfinite([u,v]).all() or np.linalg.norm(u)<1e-9:
        raise ValueError('Finite nonzero anatomical rays required')
    u=u/np.linalg.norm(u);v=v-u*np.dot(u,v)
    if np.linalg.norm(v)<1e-9:raise ValueError('Finger and thumb rays must not be collinear')
    v=v/np.linalg.norm(v)
    return np.stack([u,v,np.cross(u,v)],axis=1)


def calibration(robot):
    import torch
    path=OUT/'vendor/kimodo/kimodo/assets/skeletons/somaskel77/joints.p'
    neutral=torch.load(path,map_location='cpu',weights_only=True).squeeze().numpy()
    names=[n for n,_ in skeleton_names(77)];rest=robot.fk(np.zeros(len(robot.names)))
    result={}
    for side,prefix in [('l','Left'),('r','Right')]:
        source=anatomical_basis(neutral[names.index(prefix+'HandMiddleEnd')]-neutral[names.index(prefix+'Hand')],
            neutral[names.index(prefix+'HandThumbEnd')]-neutral[names.index(prefix+'Hand')])
        origin=rest['hand_'+side][:3,3]
        world=anatomical_basis(rest['middle3_'+side][:3,3]-origin,rest['thumb3_'+side][:3,3]-origin)
        local=rest['hand_'+side][:3,:3].T@world
        result[side]={'source_neutral_basis':source,'target_hand_local_basis':local}
    return result


def refine(parent,run):
    r=Robot();cal=calibration(r);src,sk=load_source(run/'source.npz');ix={n:i for i,(n,_) in enumerate(sk)}
    with np.load(parent/'target.npz') as data:d={k:data[k] for k in data.files}
    ret=json.loads((parent/'retarget.json').read_text());C=np.asarray(ret['source_coordinate_matrix'])
    original=d['q'];base=d['base'];n=len(original)
    active=[i for i,name in enumerate(r.names) if 'forearm_roll' in name or 'wrist_pitch' in name]
    desired=np.stack([np.einsum('ij,tjk,kl->til',C,src['global_rot_mats'][:,ix[prefix+'Hand']],cal[side]['source_neutral_basis'])
                      for side,prefix in [('l','Left'),('r','Right')]],axis=1)
    def axes(q,f):
        tf=r.fk(q,base[f]);return np.stack([tf['hand_'+side][:3,:3]@cal[side]['target_hand_local_basis'] for side in ('l','r')])
    before=np.array([axes(q,f) for f,q in enumerate(original)]);corrected=original.copy();success=[]
    for f in range(n):
        previous=np.zeros(len(active)) if f==0 else corrected[f-1,active]-original[f-1,active]
        def residual(x):
            q=original[f].copy();q[active]=x
            return np.r_[.05*(axes(q,f)-desired[f]).ravel(),.002*(x-original[f,active]),.003*(x-original[f,active]-previous)]
        initial=np.clip(original[f,active],r.lower[active],r.upper[active])
        # Legacy position-only QP leaves ~1e-15 values. A nonzero but tiny
        # initial norm makes TRF's first trust region tiny and triggers ftol
        # before a meaningful orientation step. Exact neutral uses radius1.
        initial[np.abs(initial)<1e-8]=0.
        fit=least_squares(residual,initial,
            bounds=(r.lower[active],r.upper[active]),max_nfev=25,ftol=1e-6)
        corrected[f,active]=fit.x;success.append(bool(fit.success))
    delta=gaussian_filter1d(corrected[:,active]-original[:,active],.6,axis=0,mode='nearest')
    corrected[:,active]=np.clip(original[:,active]+delta,r.lower[active],r.upper[active])
    after=np.array([axes(q,f) for f,q in enumerate(corrected)])
    poses=[r.fk(q,b) for q,b in zip(corrected,base)]
    fitted=np.array([[tf[t][:3,3] for _,t,_ in MAP] for tf in poses])
    drift=float(np.max(np.linalg.norm(fitted-d['fitted'],axis=-1)))
    if drift>1e-5:raise ValueError(f'Unexpected wrist-position effect: {drift}m')
    per_frame={};stats={}
    for k,side in enumerate(('l','r')):
        for j,label in enumerate(('finger','thumb_side','palm_normal')):
            for version,values in [('before',before),('after',after)]:
                key=f'{side}_{label}_{version}_error_deg'
                angle=np.degrees(np.arccos(np.clip(np.sum(values[:,k,:,j]*desired[:,k,:,j],axis=-1),-1,1)))
                per_frame[key]=angle.tolist();stats[key]=summary(angle,d['times'],25.)
    report={'purpose':'Animation orientation diagnostic; fingers remain locked. No anatomy reflection.',
        'method':'Middle-finger ray plus orthogonal thumb-side ray define source/target proper anatomical bases. Track C R_source_hand B_source using target_hand_R B_target_local.',
        'source_neutral_sha256':sha256(OUT/'vendor/kimodo/kimodo/assets/skeletons/somaskel77/joints.p'),
        'calibration':{s:{k:v.tolist() for k,v in a.items()} for s,a in cal.items()},
        'active_joints':[r.names[i] for i in active],'max_landmark_position_change_m':drift,
        'upper_arm_and_leg_joints_unchanged':True,'root_unchanged':True,'optimizer_success':success,
        'statistics':stats,'per_frame':per_frame,'frame_count':n,
        'limitations':['Tracks neutral finger/palm orientation driven by source hand rotation; does not animate source finger articulation.',
            'Two available revolutes per hand cannot match every3D hand orientation exactly within canonical joint ranges.']}
    d.update(q=corrected,fitted=fitted,hand_correction_q=corrected-original)
    np.savez_compressed(run/'target.npz',**d);write_json(run/'hands.json',report)
    ret['hand_orientation_refinement']={'active_joints':report['active_joints'],'max_landmark_position_change_m':drift,'orientation_weight_m':.05,'pose_prior_m_rad':.002,'temporal_correction_prior_m_rad':.003,'correction_sigma_frames':.6}
    write_json(run/'retarget.json',ret)
    return report


def render_hands(parent,run):
    from PIL import Image,ImageDraw,ImageFont
    from render import FONT,encoder_command,arrow
    from review import decode,sheets
    r=Robot();cal=calibration(r);src,sk=load_source(run/'source.npz');ix={n:i for i,(n,_) in enumerate(sk)}
    C=np.asarray(json.loads((run/'retarget.json').read_text())['source_coordinate_matrix'])
    clips=[]
    for p in (parent,run):
        with np.load(p/'target.npz') as data:clips.append((data['q'],data['base']))
    n=len(clips[0][0]);font=ImageFont.truetype(FONT,15)
    proc=subprocess.Popen(encoder_command(run/'hands_comparison.mp4',960,720,30),stdin=subprocess.PIPE)
    try:
        for f in range(n):
            im=Image.new('RGB',(960,720),'#eff3f5');draw=ImageDraw.Draw(im)
            draw.text((10,6),f'Hand orientation | frame{f} | {f/30:.3f}s | green=source finger, red=target finger | fixed XZ view',font=font,fill='#263e4d')
            for row,(q,base) in enumerate(clips):
                tf=r.fk(q[f],base[f])
                for col,(side,prefix) in enumerate([('l','Left'),('r','Right')]):
                    ox=col*480;oy=32+row*342;center=tf['hand_'+side][:3,3]
                    def project(v):
                        v=np.asarray(v);return np.stack([ox+240+(v[...,0]-center[0])*1300,oy+175-(v[...,2]-center[2])*1300],axis=-1)
                    draw.rectangle((ox+2,oy,ox+478,oy+338),outline='#acbdc9')
                    draw.text((ox+10,oy+6),('BEFORE' if row==0 else 'AFTER')+' | '+prefix+' hand | wrist centered',font=font,fill='#29444f')
                    polygons=[]
                    for link,v,faces in r.vertices(tf):
                        if not (link.endswith('_'+side) and any(k in link for k in ('hand','thumb','index','middle','ring','pinky'))):continue
                        points=project(v)
                        for face,depth in zip(faces,-v[faces,1].mean(1)):polygons.append((depth,points[face]))
                    for _,points in sorted(polygons,key=lambda x:x[0]):draw.polygon([tuple(p) for p in points],fill='#73a1b8' if side=='l' else '#b882a1',outline='#5d7180')
                    desired=C@src['global_rot_mats'][f,ix[prefix+'Hand']]@cal[side]['source_neutral_basis']
                    actual=tf['hand_'+side][:3,:3]@cal[side]['target_hand_local_basis']
                    arrow(draw,project(center),project(center+.10*desired[:,0]),'#278d43','source',font)
                    arrow(draw,project(center),project(center+.10*actual[:,0]),'#b54946','target',font)
            proc.stdin.write(im.tobytes())
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('Hand comparison encoding failed')
    frames=decode(run/'hands_comparison.mp4');report=json.loads((run/'hands.json').read_text())
    selected=set(np.linspace(0,n-1,min(24,n),dtype=int).tolist())
    for value in report['statistics'].values():selected.add(value['worst_frame'])
    folder=run/'hand_review';folder.mkdir(exist_ok=True)
    report.update(review_frames=sorted(selected),review_pages=sheets(frames,sorted(selected),folder,'hands'),
                  video_sha256=sha256(run/'hands_comparison.mp4'),visual_review='pending')
    write_json(run/'hands.json',report)


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-run',required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    for value in (a.parent_run,a.run_id):
        if not value.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run ID')
    parent=OUT/'runs'/a.parent_run;run=OUT/'runs'/a.run_id;run.mkdir(exist_ok=False)
    for name in ('source.npz','source_validation.json'):shutil.copyfile(parent/name,run/name)
    shutil.copyfile(parent/'target.npz',run/'target_before_hands.npz')
    metadata=json.loads((parent/'source_metadata.json').read_text());metadata.update(retarget_parent=a.parent_run,
        hand_refine_command=__import__('sys').argv,hand_refine_sha256=sha256(__file__),
        hand_refine_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write_json(run/'source_metadata.json',metadata)
    progress('hand_orientation_refinement',active_run=run.name,active_process='CPU forearm/wrist IK',next_step='Measure hand orientation before/after and inspect actual mesh motion')
    report=refine(parent,run)
    from validate import validate_file,compare_semantics
    from quality import measure
    from render import render
    v=validate_file(run,'composite');compare_semantics(run);measure(run);render(run,'Calibrated hand orientation | '+run.name);render_hands(parent,run)
    v['clip_status']='generated_and_rendered';write_json(run/'validation.json',v)
    arts=[artifact(run/n,'video' if n.endswith('.mp4') else 'json') for n in ['hands.json','hands_comparison.mp4','clean_preview.mp4','proof.mp4','quality_frames.json','validation.json']]
    node(run.name+':retargeted',[parent.name+':retargeted'],'passed',label='Calibrated wrist/forearm orientation',artifacts=arts)
    node(run.name+':validated',[run.name+':retargeted'],'passed',label='Hand refinement rendered; review pending',artifacts=arts)
    progress('hand_variant_rendered',active_run=run.name,active_process=None,metrics=v['metrics'],config=metadata,artifacts=arts,next_step='Same-source visual review; no automatic promotion')
    print(json.dumps({k: {s:v.get(s) for s in ['mean','p95','maximum']} for k,v in report['statistics'].items()}))


if __name__=='__main__':main()

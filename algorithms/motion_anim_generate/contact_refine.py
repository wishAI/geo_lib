"""Source-aware stance correction on retained animation, with original motion preserved."""
import argparse
import json
import shutil
import subprocess
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from landau import Robot
from quality import intervals, measure
from retarget import MAP, load_source, source_foot_contacts, orientation_axes
from state import ROOT,OUT,sha256,write_json,progress,node,artifact


def stance_targets(target,source,contact,ramp=3):
    """Keep actual source translation; remove only target-added stance drift.

    Each interval is anchored at its least-squares mean target position. Soft
    ramps avoid snapping at generated contact-label transitions. Z stays at the
    original animation height, so heel lift/toe-off are not flattened.
    """
    desired=np.asarray(target).copy();weight=np.zeros(len(target))
    for a,b in intervals(contact):
        if b-a<2:continue
        offset=np.mean(target[a:b+1,:2]-source[a:b+1,:2],axis=0)
        desired[a:b+1,:2]=source[a:b+1,:2]+offset
        for f in range(a,b+1):
            # Clip boundaries are not observed contact transitions. Ramping a
            # stance that spans the whole clip creates artificial start/end drift.
            entering=1. if a==0 else (f-a+1)/ramp
            leaving=1. if b==len(target)-1 else (b-f+1)/ramp
            weight[f]=min(1.,entering,leaving)
    return desired,weight


def refine(parent,run):
    r=Robot();src,sk=load_source(run/'source.npz');ix={s:i for i,(s,_) in enumerate(sk)}
    with np.load(parent/'target.npz') as d:data={k:d[k] for k in d.files}
    shutil.copyfile(parent/'target.npz',run/'target_before_contact.npz')
    ret=json.loads((parent/'retarget.json').read_text());C=np.asarray(ret['source_coordinate_matrix'])
    source=src['posed_joints']@C.T*ret['root_trajectory_scale']
    original=data['q'];base=data['base'].copy();n=len(original)
    contacts=source_foot_contacts(src['foot_contacts']);axes=orientation_axes(r)
    legs=[i for i,name in enumerate(r.names) if any(k in name for k in ('hip_','knee','shin_','ankle_','toe_joint'))]
    links=['foot_l','toes_01_l','foot_r','toes_01_r'];source_names=['LeftFoot','LeftToeBase','RightFoot','RightToeBase']
    poses=[r.fk(q,b) for q,b in zip(original,base)]
    points=np.array([[tf[t][:3,3] for t in links] for tf in poses])
    initial_axes=np.array([[tf[t][:3,:3]@axes[t] for t in links] for tf in poses])
    targets=[];weights=[]
    for k,(link,s) in enumerate(zip(links,source_names)):
        goal,w=stance_targets(points[:,k],source[:,ix[s]],contacts[:,k//2]);targets.append(goal);weights.append(w)
    targets=np.stack(targets,axis=1);weights=np.stack(weights,axis=1)
    parent_joint={j['child']:j for j in r.joints};ancestors={}
    for link in links:
        current=link;chain=[]
        while current in parent_joint:
            j=parent_joint[current]
            if j['name'] in r.index:chain.append(r.index[j['name']])
            current=j['parent']
        ancestors[link]=chain
    corrected=original.copy();success=[]
    for f in range(n):
        previous_delta=np.zeros(len(legs)) if f==0 else corrected[f-1,legs]-original[f-1,legs]
        w=1+5*weights[f]
        goal=points[f]+weights[f,:,None]*(targets[f]-points[f])
        def objective(x,jac=False):
            q=original[f].copy();q[legs]=x;tf=r.fk(q,base[f])
            pos=np.array([tf[t][:3,3] for t in links]);orient=np.array([tf[t][:3,:3]@axes[t] for t in links])
            if not jac:
                return np.r_[((pos-goal)*w[:,None]).ravel(),(.12*(orient-initial_axes[f])).ravel(),
                             .012*(x-original[f,legs]),.012*(x-original[f,legs]-previous_delta)]
            J=np.zeros((12,len(legs)));O=np.zeros((24,len(legs)))
            for col,qi in enumerate(legs):
                j=r.moving[qi];jt=tf[j['parent']]@j['origin'];axis=jt[:3,:3]@(j['axis']/np.linalg.norm(j['axis']))
                for k,t in enumerate(links):
                    if qi in ancestors[t]:
                        J[k*3:k*3+3,col]=w[k]*np.cross(axis,pos[k]-jt[:3,3])
                        O[k*6:k*6+6,col]=.12*np.cross(axis,orient[k].T).T.ravel()
            return np.vstack([J,O,.012*np.eye(len(legs)),.012*np.eye(len(legs))])
        result=least_squares(objective,np.clip(original[f,legs],r.lower[legs],r.upper[legs]),jac=lambda x:objective(x,True),
            bounds=(r.lower[legs],r.upper[legs]),max_nfev=30,ftol=1e-6)
        corrected[f,legs]=result.x;success.append(bool(result.success))
    # Smooth only the correction. Original source timing and motion are retained.
    correction=gaussian_filter1d(corrected[:,legs]-original[:,legs],sigma=.8,axis=0,mode='nearest')
    corrected[:,legs]=np.clip(original[:,legs]+correction,r.lower[legs],r.upper[legs])
    fitted=[];minz=[]
    for q,b in zip(corrected,base):
        tf=r.fk(q,b);fitted.append([tf[t][:3,3] for _,t,_ in MAP]);minz.append(min(v[:,2].min() for _,v,_ in r.vertices(tf)))
    # A single constant clearance translation avoids introducing root-height jitter.
    lift=max(0.,.001-min(minz));base[:,2,3]+=lift;fitted=np.asarray(fitted);fitted[:,:,2]+=lift
    errors=np.linalg.norm(fitted-data['desired'],axis=-1)
    data.update(q=corrected,base=base,fitted=fitted,errors_m=errors,contact_correction_q=corrected-original,
                stance_position_targets=targets,stance_target_weights=weights)
    np.savez_compressed(run/'target.npz',**data)
    ret['before_contact_metrics']={k:ret[k] for k in ('retarget_rmse_m','max_landmark_error_m')}
    ret.update(unadjusted_reference_rmse_m=float(np.sqrt(np.mean(np.sum((fitted-data['unadjusted_desired'])**2,axis=-1)))),
        retarget_rmse_m=float(np.sqrt(np.mean(errors**2))),max_landmark_error_m=float(errors.max()),
        per_landmark_rmse_m={t:float(np.sqrt(np.mean(errors[:,k]**2))) for k,(_,t,_) in enumerate(MAP)},
        source_aware_contact={'method':'Per-stance source-displacement ankle/toe targets; leg IK; retained foot orientation; smooth correction only',
            'contact_position_weight':'1 + 5*boundary-ramped source contact','orientation_weight_m':.12,
            'pose_prior_weight_m_rad':.012,'correction_smoothing_sigma_frames':.8,'max_nfev':30,
            'constant_clearance_lift_m':lift,'joint_correction_rms_rad':float(np.sqrt(np.mean((corrected-original)**2))),
            'joint_correction_max_rad':float(np.max(np.abs(corrected-original))),'optimizer_success':success,
            'pose_fidelity_tradeoff':'Leg positions change to reduce added stance drift; root rotation and upper-body joints retained. No flattening of source swing orientation/height.'})
    write_json(run/'retarget.json',ret)
    return ret


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-run',required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    for n in [a.parent_run,a.run_id]:
        if not n.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run ID')
    parent=OUT/'runs'/a.parent_run;run=OUT/'runs'/a.run_id;run.mkdir(exist_ok=False)
    for name in ('source.npz','source_validation.json'):shutil.copyfile(parent/name,run/name)
    meta=json.loads((parent/'source_metadata.json').read_text());meta.update(retarget_parent=a.parent_run,
        contact_refine_command=__import__('sys').argv,contact_refine_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        contact_refine_sha256=sha256(__file__))
    write_json(run/'source_metadata.json',meta)
    progress('source_aware_contact_refinement',active_run=a.run_id,active_process='bounded CPU leg IK',next_step='Compare every frame to the same retained source and parent target')
    ret=refine(parent,run)
    from validate import validate_file,compare_semantics
    val=validate_file(run,'composite');compare_semantics(run);quality=measure(run)
    from render import render
    render(run,'Source-aware contact refinement | '+a.run_id)
    val['clip_status']='generated_and_rendered';write_json(run/'validation.json',val)
    arts=[artifact(run/n,'video' if n.endswith('.mp4') else 'json') for n in ['clean_preview.mp4','proof.mp4','quality_frames.json','validation.json','retarget.json']]
    node(a.run_id+':retargeted',[a.parent_run+':retargeted'],'passed',label='Source-aware stance correction',artifacts=arts,metrics=val['metrics'])
    node(a.run_id+':validated',[a.run_id+':retargeted'],'passed',label='Rendered; exhaustive quality review pending',artifacts=arts)
    progress('contact_variant_rendered',active_process=None,active_run=a.run_id,artifacts=arts,next_step='Dense-frame, transition and foot-closeup review; compare multiple seeds before promotion')
    print(json.dumps({'run':a.run_id,'metrics':val['metrics'],'left_slide_error':quality['statistics']['l_ankle_stance_slide_error_m_s']}))


if __name__=='__main__':main()

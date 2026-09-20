"""Every-frame animation measurements and explicit review intervals, never control gates."""
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation
from directions import diagnose
from landau import Robot
from retarget import MAP, load_source, source_foot_contacts
from state import OUT, sha256, write_json


def intervals(mask):
    """Inclusive frame intervals, including first/last-frame runs."""
    edges=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(int))
    return list(zip(np.where(edges==1)[0].tolist(),(np.where(edges==-1)[0]-1).tolist()))


def summary(values,times,threshold=None,mask=None):
    values=np.asarray(values,dtype=float);times=np.asarray(times)
    use=np.ones(len(values),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    use &= np.isfinite(values)
    indices=np.flatnonzero(use)
    if not len(indices):return {'sample_count':0,'mean':None,'p95':None,'maximum':None,'worst_frame':None,'worst_time_s':None,'flagged_intervals':[]}
    worst=int(indices[np.argmax(values[indices])]);v=values[use]
    bad=use&(values>threshold) if threshold is not None else np.zeros(len(values),dtype=bool)
    return {'sample_count':len(indices),'mean':float(v.mean()),'p50':float(np.percentile(v,50)),
            'p95':float(np.percentile(v,95)),'p99':float(np.percentile(v,99)),
            'minimum':float(v.min()),'maximum':float(v.max()),'worst_frame':worst,'worst_time_s':float(times[worst]),
            'review_threshold':threshold,'flagged_frames':int(bad.sum()),
            'flagged_intervals':[{'first_frame':a,'last_frame':b,'start_s':float(times[a]),'end_s':float(times[b])} for a,b in intervals(bad)]}


def stance_displacement(target,source,contacts):
    """Drift relative to the source's actual movement, resetting at each stance."""
    residual=np.zeros(len(target));raw=np.zeros(len(target));source_drift=np.zeros(len(target))
    for a,b in intervals(contacts):
        dt=target[a:b+1,:2]-target[a,:2];ds=source[a:b+1,:2]-source[a,:2]
        residual[a:b+1]=np.linalg.norm(dt-ds,axis=-1)
        raw[a:b+1]=np.linalg.norm(dt,axis=-1);source_drift[a:b+1]=np.linalg.norm(ds,axis=-1)
    return residual,raw,source_drift


def measure(run):
    directions=diagnose(run);ret=json.loads((run/'retarget.json').read_text())
    src,sk=load_source(run/'source.npz');ix={name:i for i,(name,_) in enumerate(sk)}
    with np.load(run/'target.npz') as d:data={k:d[k] for k in d.files}
    times=data['times'];n=len(times);dt=float(np.median(np.diff(times)))
    r=Robot();target=[];minimum=[]
    for q,base in zip(data['q'],data['base']):
        tf=r.fk(q,base);target.append([tf[t][:3,3] for _,t,_ in MAP])
        minimum.append(min(v[:,2].min() for _,v,_ in r.vertices(tf)))
    target=np.asarray(target);names=[t for _,t,_ in MAP];ti={v:i for i,v in enumerate(names)}
    C=np.asarray(ret['source_coordinate_matrix']);scale=ret['root_trajectory_scale']
    source=src['posed_joints']@C.T*scale;contact=source_foot_contacts(src['foot_contacts'])
    series={};stats={};masks={}
    def add(name,values,threshold=None,mask=None):
        series[name]=np.asarray(values,dtype=float).tolist();stats[name]=summary(values,times,threshold,mask)
        if mask is not None:masks[name]=np.asarray(mask,dtype=bool).tolist()
    for part in ('Hips','Chest','Head'):
        add(part+'_facing_error_deg',[f[part+'_forward_angle_deg'] for f in directions['per_frame']],15.)
    for side,k,prefix in [('l',0,'Left'),('r',1,'Right')]:
        foot=[f['feet'][side] for f in directions['per_frame']]
        for metric in ('forward_error_deg','sole_normal_error_deg'):
            add(side+'_'+metric,[f[metric] for f in foot],12.)
            add(side+'_stance_'+metric,[f[metric] for f in foot],12.,contact[:,k])
        for metric in ('heel_min_z_m','toe_min_z_m','sole_tilt_deg','source_sole_tilt_deg','sole_forward_pitch_deg','source_forward_pitch_deg'):
            add(side+'_'+metric,[f[metric] for f in foot])
        add(side+'_yaw_error_deg',np.abs([f['yaw_error_deg'] for f in foot]),12.)
        for label,target_name,source_name in [('ankle','foot_'+side,prefix+'Foot'),('toe','toes_01_'+side,prefix+'ToeBase')]:
            tp=target[:,ti[target_name]];sp=source[:,ix[source_name]]
            drift,raw,sd=stance_displacement(tp,sp,contact[:,k])
            consecutive=np.r_[False,contact[1:,k]&contact[:-1,k]]
            speed=np.r_[0,np.linalg.norm(np.diff(tp[:,:2],axis=0),axis=1)/dt]
            source_speed=np.r_[0,np.linalg.norm(np.diff(sp[:,:2],axis=0),axis=1)/dt]
            residual_speed=np.r_[0,np.linalg.norm(np.diff(tp[:,:2]-sp[:,:2],axis=0),axis=1)/dt]
            stem=side+'_'+label
            add(stem+'_stance_drift_error_m',drift,.012,contact[:,k])
            add(stem+'_stance_drift_m',raw,None,contact[:,k]);add(stem+'_source_stance_drift_m',sd,None,contact[:,k])
            add(stem+'_stance_slide_error_m_s',residual_speed,.08,consecutive)
            add(stem+'_stance_slide_m_s',speed,None,consecutive);add(stem+'_source_slide_m_s',source_speed,None,consecutive)
            # Root-relative swing excursion measures whether intended foot lift was flattened.
            relative_target=tp[:,2]-target[:,0,2];relative_source=sp[:,2]-source[:,ix['Hips'],2]
            reference=np.median(relative_target-relative_source)
            add(stem+'_swing_height_change_error_m',np.abs(relative_target-relative_source-reference),.02,~contact[:,k])
    for s,t,parent in MAP[1:]:
        tp=target[:,ti[t]]-target[:,ti[dict((a,b) for a,b,_ in MAP)[parent]]]
        sp=source[:,ix[s]]-source[:,ix[parent]]
        cosine=np.sum(tp*sp,axis=-1)/np.maximum(np.linalg.norm(tp,axis=-1)*np.linalg.norm(sp,axis=-1),1e-12)
        add(t+'_bone_direction_error_deg',np.degrees(np.arccos(np.clip(cosine,-1,1))),25.)
    add('landmark_rmse_m',np.sqrt(np.mean(np.sum((target-data['desired'])**2,axis=-1),axis=-1)))
    add('minimum_mesh_z_m',minimum)
    add('penetration_depth_m',np.maximum(-np.array(minimum),0),.005)
    add('joint_step_max_rad',np.r_[0,np.max(np.abs(np.diff(data['q'],axis=0)),axis=1)],.35)
    add('joint_jitter_max_rad',np.r_[np.zeros(3),np.max(np.abs(np.diff(data['q'],n=3,axis=0)),axis=1)],.12)
    add('root_step_m',np.r_[0,np.linalg.norm(np.diff(data['base'][:,:3,3],axis=0),axis=1)],.08)
    root_rot=Rotation.from_matrix(data['base'][:,:3,:3])
    add('root_angle_step_rad',np.r_[0,(root_rot[:-1].inv()*root_rot[1:]).magnitude()],.35)
    transitions=np.where(np.any(contact[1:]!=contact[:-1],axis=1))[0]+1
    turn=np.asarray(series['root_angle_step_rad']);turn_frames=np.argsort(turn)[-4:]
    selected=set(np.linspace(0,n-1,min(n,24),dtype=int).tolist())
    for s in stats.values():
        if s['worst_frame'] is not None:selected.add(s['worst_frame'])
    for frame in np.r_[transitions,turn_frames]:
        selected.update(range(max(0,int(frame)-1),min(n,int(frame)+2)))
    report={'schema_version':1,'purpose':'every-frame animation diagnostics; thresholds select review frames, not acceptance gates',
            'frame_count':n,'duration_s':n*dt,'source_sha256':sha256(run/'source.npz'),'target_sha256':sha256(run/'target.npz'),
            'implementation_sha256':sha256(__file__),'statistics':stats,'per_frame':series,'measurement_masks':masks,
            'source_contact':contact.tolist(),'contact_transition_frames':transitions.tolist(),'turn_review_frames':turn_frames.tolist(),
            'review_frames':sorted(selected),'evenly_spaced_review_frames':np.linspace(0,n-1,min(n,24),dtype=int).tolist(),
            'limitations':['Contact labels are generated source estimates; ankle/toe stance measures are pivot proxies, not exact deforming sole contacts.',
                'Sliding error subtracts actual scaled source movement; raw source/target sliding is also retained.',
                'Joint positions and bone directions do not establish complete surface/axial-pose fidelity.',
                'Full-duration playback requires visual inspection; numeric scores and decoded frames alone do not certify visual quality.']}
    write_json(run/'quality_frames.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');a=p.parse_args()
    for name in a.runs:
        d=measure(OUT/'runs'/name)
        print(json.dumps({'run':name,'frames':d['frame_count'],'review_frames':len(d['review_frames']),
                         'left_slide_error':d['statistics']['l_ankle_stance_slide_error_m_s'],
                         'right_slide_error':d['statistics']['r_ankle_stance_slide_error_m_s']}))

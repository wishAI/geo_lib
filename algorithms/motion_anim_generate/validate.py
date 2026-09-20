"""Kinematic screening only. Mesh floor is exact; contacts/capsules/semantics are heuristics."""
import numpy as np
from scipy.spatial.transform import Rotation
from landau import Robot
from retarget import MAP
from state import write_json

TOLERANCES = dict(time_step_s=1e-5, quaternion_norm=1e-4, quaternion_step_rad=.35,
                  joint_limit_rad=1e-5, locked_joint_rad=1e-6, joint_acceleration_rad_s2=80.,
                  root_step_m=.08, foot_contact_height_m=.012, floor_penetration_m=.005,
                  foot_sliding_m_s=.08, retarget_rmse_m=.03, retarget_peak_m=.08,
                  idle_drift_m=.04, walk_forward_m=.15, walk_foot_excursion_m=.015,
                  turn_angle_rad=np.pi/4, turn_drift_m=.35, wave_height_above_chest_m=.04,
                  wave_hand_excursion_m=.04, wave_direction_changes=2)


def segment_distance(a,b,c,d):
    # Closest points between closed segments, including parallel/degenerate cases.
    u=b-a;v=d-c;w=a-c;A=u@u;B=u@v;D=v@v;E=u@w;F=v@w
    candidates=[]
    for s in (0.,1.):
        t=np.clip((F+s*B)/max(D,1e-15),0,1);candidates.append(np.linalg.norm(w+s*u-t*v))
    for t in (0.,1.):
        s=np.clip((t*B-E)/max(A,1e-15),0,1);candidates.append(np.linalg.norm(w+s*u-t*v))
    det=A*D-B*B
    if det>1e-15:
        s=(B*F-D*E)/det;t=(A*F-B*E)/det
        if 0<=s<=1 and 0<=t<=1:candidates.append(np.linalg.norm(w+s*u-t*v))
    return min(candidates)


def semantics(action, positions, names, yaw, times, thresholds=TOLERANCES):
    ix={n:i for i,n in enumerate(names)};root=positions[:,ix['root_x']]
    heading=np.array([-np.sin(yaw[0]),np.cos(yaw[0])])
    delta=root[-1,:2]-root[0,:2]
    drift=float(np.max(np.linalg.norm(root[:,:2]-root[0,:2],axis=1)))
    forward=float(delta@heading);yaw_change=float(yaw[-1]-yaw[0])
    metrics={'forward_displacement_m':forward,'root_drift_m':drift,'yaw_change_rad':yaw_change}
    if action=='idle':
        good=drift<=thresholds['idle_drift_m'] and abs(yaw_change)<.25
    elif action=='walk':
        feet=positions[:,[ix['foot_l'],ix['foot_r']]]-root[:,None]
        excursion=float(np.max(np.ptp(feet[:,:,:2],axis=0)))
        metrics['relative_foot_excursion_m']=excursion
        good=forward>=thresholds['walk_forward_m'] and excursion>=thresholds['walk_foot_excursion_m'] and abs(yaw_change)<.6
    elif action=='turn':
        good=abs(yaw_change)>=thresholds['turn_angle_rad'] and drift<=thresholds['turn_drift_m']
    elif action=='wave':
        results=[]
        for side in ('l','r'):
            hand=positions[:,ix['hand_'+side]];chest=positions[:,ix['spine_03_x']]
            relative=hand-chest
            raised=relative[:,2]>thresholds['wave_height_above_chest_m']
            # Ignore tiny velocity signs and require actual oscillation while raised.
            velocity=np.diff(relative[:,0]);mask=raised[1:] & raised[:-1] & (np.abs(velocity)>.0008)
            signs=np.sign(velocity[mask]);changes=int(np.sum(signs[1:]!=signs[:-1]))
            excursion=float(np.ptp(relative[raised,0])) if np.any(raised) else 0.
            results.append(bool(raised.mean()>.2 and changes>=thresholds['wave_direction_changes'] and excursion>=thresholds['wave_hand_excursion_m']))
            metrics[f'{side}_raised_fraction']=float(raised.mean());metrics[f'{side}_direction_changes']=changes
        good=any(results) and drift<=.15
    else:
        return {'status':'unavailable','reason':'No single-action claim for composite or unconditional source','metrics':metrics,'heuristic':True}
    return {'status':'passed' if good else 'failed','metrics':metrics,'heuristic':True}


def validate(data, robot=None, action='idle', thresholds=None):
    r=robot or Robot();tol={**TOLERANCES,**(thresholds or {})};violations=[]
    report={'schema_version':1,'kinematic_pass':False,'dynamic_feasibility':'not tested',
            'robot_control_safety':'not established; no actuation', 'thresholds':tol,
            'heuristic_checks':['contact labels from target geometry','capsule self-intersection proxy','motion semantics'],
            'unavailable_checks':['exact triangle self-intersection','balance, forces, torque and actuator tracking','source-to-target axial orientation fidelity'],
            'violations':violations}
    def add(code, frame, **extra):violations.append({'check':code,'frame':int(frame),**extra})
    required=['q','base','base_quat_xyzw','times','errors_m']
    for key in required:
        if key not in data:
            add('missing_field',-1,field=key);return report
        arr=np.asarray(data[key])
        if not np.isfinite(arr).all():
            bad=np.unique(np.argwhere(~np.isfinite(arr))[:,0]) if arr.ndim else [-1]
            for f in bad:add('nonfinite',f,field=key)
    if violations:return report
    q=np.asarray(data['q']);times=np.asarray(data['times']);bases=np.asarray(data['base']);quat=np.asarray(data['base_quat_xyzw'])
    n=len(times)
    if n<3 or q.shape!=(n,len(r.names)) or bases.shape!=(n,4,4) or quat.shape!=(n,4):
        add('shape_or_duration',-1);return report
    if 'joint_names' not in data or list(data['joint_names'])!=r.names:
        add('joint_order',-1);return report
    dt=np.diff(times)
    if np.any(dt<=0) or np.max(np.abs(dt-np.median(dt)))>tol['time_step_s']:
        for f in np.where((dt<=0)|(np.abs(dt-np.median(dt))>tol['time_step_s']))[0]:add('time_continuity',f+1)
        return report
    h=float(np.median(dt));speed=np.diff(q,axis=0)/dt[:,None];acc=np.diff(speed,axis=0)/h
    tests=[('joint_limits',(q<r.lower-tol['joint_limit_rad'])|(q>r.upper+tol['joint_limit_rad']),0),
           ('joint_speed',np.abs(speed)>r.speed+1e-6,1),
           ('joint_acceleration',np.abs(acc)>tol['joint_acceleration_rad_s2'],2)]
    for code,mask,shift in tests:
        for f,j in np.argwhere(mask):add(code,f+shift,joint=r.names[j])
    for f,j in np.argwhere(np.abs(q[:,r.locked])>tol['locked_joint_rad']):add('locked_joint',f,joint=r.names[r.locked[j]])
    norms=np.linalg.norm(quat,axis=1)
    for f in np.where(np.abs(norms-1)>tol['quaternion_norm'])[0]:add('quaternion_norm',f)
    if np.any(norms<1e-9):return report
    normalized=quat/norms[:,None];dots=np.sum(normalized[1:]*normalized[:-1],axis=1)
    for f in np.where(dots<0)[0]:add('quaternion_sign_discontinuity',f+1)
    angles=2*np.arccos(np.clip(np.abs(dots),0,1))
    for f in np.where(angles>tol['quaternion_step_rad'])[0]:add('quaternion_step',f+1)
    for f in np.where(np.linalg.norm(np.diff(bases[:,:3,3],axis=0),axis=1)>tol['root_step_m'])[0]:add('root_discontinuity',f+1)
    expected=Rotation.from_quat(normalized).as_matrix()
    for f in np.where(np.max(np.abs(expected-bases[:,:3,:3]),axis=(1,2))>1e-4)[0]:add('base_rotation_mismatch',f)
    for f in np.where(np.max(np.abs(bases[:,3]-[0,0,0,1]),axis=1)>1e-8)[0]:add('base_homogeneous_row',f)
    positions=[];min_z=[];foot_centers=[];foot_bottom=[];proxy=[]
    capsules=[('thigh_stretch_l','leg_stretch_l',.022),('thigh_stretch_r','leg_stretch_r',.022),
              ('leg_stretch_l','foot_l',.018),('leg_stretch_r','foot_r',.018),
              ('arm_stretch_l','forearm_stretch_l',.016),('arm_stretch_r','forearm_stretch_r',.016),
              ('forearm_stretch_l','hand_l',.014),('forearm_stretch_r','hand_r',.014),
              ('spine_01_x','neck_x',.035)]
    for f in range(n):
        tf=r.fk(q[f],bases[f]);positions.append([tf[t][:3,3] for t in r.links]);verts=r.vertices(tf)
        lowest=min(v[:,2].min() for _,v,_ in verts);min_z.append(lowest)
        if lowest < -tol['floor_penetration_m']:add('floor_penetration',f,depth_m=float(-lowest))
        centers=[];bottom=[]
        for side in ('l','r'):
            fv=np.concatenate([v for link,v,_ in verts if link in ('foot_'+side,'toes_01_'+side)])
            centers.append(fv.mean(0));bottom.append(fv[:,2].min())
        foot_centers.append(centers);foot_bottom.append(bottom)
        overlaps=[]
        for i,(a,b,radius) in enumerate(capsules):
            for c,d,radius2 in capsules[i+1:]:
                if {a,b}&{c,d}:continue
                depth=radius+radius2-segment_distance(tf[a][:3,3],tf[b][:3,3],tf[c][:3,3],tf[d][:3,3])
                if depth>.003:
                    overlaps.append([a,b,c,d]);add('self_intersection_proxy',f,pair=[a,b,c,d],depth_m=float(depth))
        proxy.append(overlaps)
    positions=np.array(positions);foot_centers=np.array(foot_centers);foot_bottom=np.array(foot_bottom)
    contacts=foot_bottom<=tol['foot_contact_height_m']
    vfoot=np.linalg.norm(np.diff(foot_centers[:,:,:2],axis=0),axis=-1)/h
    slide=(contacts[:-1]&contacts[1:])&(vfoot>tol['foot_sliding_m_s'])
    for f,side in np.argwhere(slide):add('foot_sliding',f+1,side=['left','right'][side],speed_m_s=float(vfoot[f,side]))
    errors=np.asarray(data['errors_m']);rmse=float(np.sqrt(np.mean(errors**2)))
    for f in np.where(np.max(errors,axis=1)>tol['retarget_peak_m'])[0]:add('retarget_peak',f)
    if rmse>tol['retarget_rmse_m']:add('retarget_rmse',-1,value_m=rmse)
    yaw=np.unwrap(np.arctan2(bases[:,1,0],bases[:,0,0]))
    sem=semantics(action,positions,r.links,yaw,times,tol)
    if sem['status']=='failed':add('motion_semantics',-1,action=action)
    report.update(kinematic_pass=not violations and sem['status']=='passed',
                  geometry_and_continuity_pass=not any(v['check']!='motion_semantics' for v in violations),
                  semantics=sem,frame_count=n,duration_s=n*h,sampled_span_s=float(times[-1]-times[0]),
                  metrics={'retarget_rmse_m':rmse,'min_mesh_z_m':float(min(min_z)),
                           'max_joint_speed_rad_s':float(np.max(np.abs(speed))),
                           'max_joint_acceleration_rad_s2':float(np.max(np.abs(acc))),
                           'sliding_frames':int(np.any(slide,axis=1).sum()),
                           'self_intersection_proxy_frames':sum(bool(x) for x in proxy)},
                  per_frame={'minimum_mesh_z_m':[float(x) for x in min_z],
                             'foot_min_z_m':foot_bottom.tolist(),'foot_contact_heuristic':contacts.tolist()})
    return report


def validate_file(run,action):
    with np.load(run/'target.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    report=validate(data,action=action);write_json(run/'validation.json',report);return report


def validate_source(path):
    from retarget import load_source
    src,skeleton=load_source(path);issues=[];p=src['posed_joints'];n=len(p)
    for key,value in src.items():
        if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():
            for f in np.unique(np.argwhere(~np.isfinite(value))[:,0]):issues.append({'frame':int(f),'check':'nonfinite','field':key})
    if 'global_rot_mats' in src:
        rot=src['global_rot_mats']
        if rot.shape!=(n,len(skeleton),3,3):issues.append({'frame':-1,'check':'rotation_shape'})
        else:
            error=np.max(np.abs(np.swapaxes(rot,-1,-2)@rot-np.eye(3)),axis=(-1,-2))
            determinant=np.linalg.det(rot)
            for f,j in np.argwhere((error>1e-3)|(np.abs(determinant-1)>1e-3)):
                issues.append({'frame':int(f),'joint':skeleton[j][0],'check':'rotation_SO3'})
    return {'finite_rotation_pass':not issues,'violations':issues,'frame_count':n,'duration_s':n/30,
            'checks':['finite source arrays','global rotation orthogonality and determinant, 1e-3 tolerance'],
            'not_checked':['source physical feasibility','source joint limits: SOMA is not Landau'],
            'frame_rate_contract_hz':30}

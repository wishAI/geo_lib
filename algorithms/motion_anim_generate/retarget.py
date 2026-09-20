"""SOMA landmarks -> copied Landau URDF using bounded, temporally regularized IK."""
import ast
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from landau import Robot
from state import ROOT, write_json

# Proper basis change: SOMA +Y up/+Z forward -> application +Z up/+Y forward.
# Canonical Landau toe geometry faces native -Y. Hip-derived base yaw near pi
# transports native forward to application +Y, preserving named left/right and
# the URDF's root_x +90-degree roll. Never reflect or swap anatomical joints.
C = np.array([[-1., 0, 0], [0, 0, 1], [0, 1, 0]])
# Ordered hierarchy for per-segment proportion adaptation.
MAP = [
    ('Hips', 'root_x', None), ('Spine1', 'spine_01_x', 'Hips'),
    ('Spine2', 'spine_02_x', 'Spine1'), ('Chest', 'spine_03_x', 'Spine2'),
    ('Neck1', 'neck_x', 'Chest'), ('Head', 'head_x', 'Neck1'),
    ('LeftArm', 'arm_stretch_l', 'Chest'), ('LeftForeArm', 'forearm_stretch_l', 'LeftArm'),
    ('LeftHand', 'hand_l', 'LeftForeArm'),
    ('RightArm', 'arm_stretch_r', 'Chest'), ('RightForeArm', 'forearm_stretch_r', 'RightArm'),
    ('RightHand', 'hand_r', 'RightForeArm'),
    ('LeftLeg', 'thigh_stretch_l', 'Hips'), ('LeftShin', 'leg_stretch_l', 'LeftLeg'),
    ('LeftFoot', 'foot_l', 'LeftShin'), ('LeftToeBase', 'toes_01_l', 'LeftFoot'),
    ('RightLeg', 'thigh_stretch_r', 'Hips'), ('RightShin', 'leg_stretch_r', 'RightLeg'),
    ('RightFoot', 'foot_r', 'RightShin'), ('RightToeBase', 'toes_01_r', 'RightFoot'),
]


def skeleton_names(count):
    # Read data literals from pinned vendor, without importing torch or executing source.
    path = ROOT/'outputs/vendor/kimodo/kimodo/skeleton/definitions.py'
    for cls in ast.parse(path.read_text()).body:
        if isinstance(cls, ast.ClassDef) and cls.name == f'SOMASkeleton{count}':
            for item in cls.body:
                if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'bone_order_names_with_parents' for t in item.targets):
                    return ast.literal_eval(item.value)
    raise ValueError(f'Unsupported source skeleton: {count}; expected SOMA 30 or 77')


def load_source(path):
    with np.load(path, allow_pickle=False) as f:
        src = {k: f[k] for k in f.files}
    p = src['posed_joints']
    if p.ndim != 3 or p.shape[0] < 3 or p.shape[2] != 3 or not np.isfinite(p).all():
        raise ValueError('Invalid source positions')
    skeleton = skeleton_names(p.shape[1])
    return src, skeleton


def source_foot_contacts(values):
    values=np.asarray(values)
    if values.ndim!=2 or values.shape[1] not in (4,6):
        raise ValueError('SOMA contact channels must be 4 (30-joint) or 6 (77-joint)')
    return values.reshape(len(values),2,values.shape[1]//2).max(-1)>.5


def orientation_axes(robot):
    """Local anatomical forward/up, calibrated from the unchanged mounted rest pose.

    Ankle-to-toe vectors include the ankle height: their vertical component is
    not a sole pitch. The rest mesh soles are approximately world-horizontal.
    """
    rest=robot.fk(np.zeros(len(robot.names)));result={}
    for side in ('l','r'):
        forward=rest['toes_01_'+side][:3,3]-rest['foot_'+side][:3,3]
        forward[2]=0;forward/=np.linalg.norm(forward)
        for link in ('foot_'+side,'toes_01_'+side):
            result[link]=rest[link][:3,:3].T@np.stack([forward,[0.,0,1]],axis=1)
    for link in ('spine_03_x','head_x'):
        result[link]=rest[link][:3,:3].T@np.array([[0.,0],[-1,0],[0,1]])
    return result


def solve(source, destination, fps=30., max_nfev=45, speed_bounded=False, anchor_rigid=False, foot_orientation=False):
    src, skeleton = load_source(source)
    names = [x[0] for x in skeleton]
    idx = {n:i for i,n in enumerate(names)}
    robot = Robot()
    q0 = np.zeros(len(robot.names))
    rest = robot.fk(q0)
    mapping = {s:t for s,t,_ in MAP}
    p = src['posed_joints'] @ C.T
    # Root trajectory scales by leg length, while limb directions use individual lengths.
    source_leg = sum(np.linalg.norm(p[0,idx[a]]-p[0,idx[b]]) for a,b in [('LeftLeg','LeftShin'),('LeftShin','LeftFoot')])
    target_leg = sum(np.linalg.norm(rest[a][:3,3]-rest[b][:3,3]) for a,b in [('thigh_stretch_l','leg_stretch_l'),('leg_stretch_l','foot_l')])
    scale = target_leg/source_leg
    root = rest['root_x'][:3,3] + (p[:,idx['Hips']]-p[0,idx['Hips']])*scale
    desired = np.zeros((len(p),len(MAP),3)); desired[:,0]=root
    for k,(s,t,parent) in enumerate(MAP[1:],1):
        parent_idx = [v[0] for v in MAP].index(parent)
        v = p[:,idx[s]]-p[:,idx[parent]]
        length = np.linalg.norm(rest[t][:3,3]-rest[mapping[parent]][:3,3])
        desired[:,k] = desired[:,parent_idx] + v / np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9)*length
    lateral = p[:,idx['LeftLeg']]-p[:,idx['RightLeg']]
    yaw = np.unwrap(np.arctan2(lateral[:,1],lateral[:,0]))
    if foot_orientation:
        if 'global_rot_mats' not in src:raise ValueError('Orientation retarget requires source global rotations')
        # SOMA neutral +Z is anatomical forward; mounted Landau native -Y is forward.
        forward=src['global_rot_mats'][:,idx['Hips'],:,2]@C.T
        yaw=np.unwrap(np.arctan2(forward[:,0],-forward[:,1]))
    # The root mount stays intact; base pose only transports its world anchor and yaw.
    bases = np.repeat(np.eye(4)[None],len(p),axis=0)
    bases[:,:3,:3]=Rotation.from_euler('z',yaw).as_matrix()
    if foot_orientation:
        bases[:,:3,:3]=np.einsum('ij,tjk,kl->til',C,src['global_rot_mats'][:,idx['Hips']],rest['root_x'][:3,:3].T)
    bases[:,:3,3]=root-np.einsum('tij,j->ti',bases[:,:3,:3],rest['root_x'][:3,3])
    unadjusted_desired=desired.copy()
    if anchor_rigid:
        # These pivot positions have zero joint-position Jacobians. Transport
        # the robot's actual pelvis attachments rather than impossible human offsets.
        for k,(s,t,parent) in enumerate(MAP[1:],1):
            if t in ('spine_01_x','thigh_stretch_l','thigh_stretch_r'):
                desired[:,k]=np.einsum('tij,j->ti',bases[:,:3,:3],rest[t][:3,3])+bases[:,:3,3]
            else:
                parent_idx=[v[0] for v in MAP].index(parent)
                v=p[:,idx[s]]-p[:,idx[parent]]
                length=np.linalg.norm(rest[t][:3,3]-rest[mapping[parent]][:3,3])
                desired[:,k]=desired[:,parent_idx]+v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9)*length
    all_q=[]; fitted=[]; iterations=[]; optimizer_success=[]
    parent_joint = {j['child']:j for j in robot.joints}
    ancestors={}
    for _,target,_ in MAP:
        cur=target; chain=[]
        while cur in parent_joint:
            j=parent_joint[cur]
            if j['name'] in robot.index: chain.append(robot.index[j['name']])
            cur=j['parent']
        ancestors[target]=chain
    active=robot.active; smooth=0.012; neutral=0.003
    axes=orientation_axes(robot) if foot_orientation else {}
    orientation_map={t:s for s,t,_ in MAP if t in axes}
    desired_axes={t:np.einsum('ij,tjk,kl->til',C,src['global_rot_mats'][:,idx[s]],
                            np.array([[0.,0],[0,1],[1,0]])) for t,s in orientation_map.items()}
    weights={t:.10 if t.startswith(('foot','toes')) else .05 for t in axes}
    previous=q0.copy()
    for frame in range(len(p)):
        base=bases[frame]
        def evaluate(x, jac=False):
            q=q0.copy();q[active]=x; tf=robot.fk(q,base)
            pred=np.array([tf[t][:3,3] for _,t,_ in MAP])
            current_axes={t:tf[t][:3,:3]@a for t,a in axes.items()}
            if not jac:
                orient=np.concatenate([(current_axes[t]-desired_axes[t][frame]).ravel()*weights[t] for t in axes]) if axes else np.empty(0)
                return np.r_[(pred-desired[frame]).ravel(),orient,smooth*(x-previous[active]),neutral*x]
            jmat=np.zeros((len(MAP)*3,len(active)))
            ojmat=np.zeros((len(axes)*6,len(active)))
            for col,qi in enumerate(active):
                j=robot.moving[qi]; joint_tf=tf[j['parent']]@j['origin']
                axis=joint_tf[:3,:3]@(j['axis']/np.linalg.norm(j['axis']))
                for row,(_,t,_) in enumerate(MAP):
                    if qi in ancestors[t]: jmat[row*3:row*3+3,col]=np.cross(axis,pred[row]-joint_tf[:3,3])
                for row,t in enumerate(axes):
                    if qi in ancestors[t]:ojmat[row*6:row*6+6,col]=np.cross(axis,current_axes[t].T).T.ravel()*weights[t]
            return np.vstack([jmat,ojmat,smooth*np.eye(len(active)),neutral*np.eye(len(active))])
        lower=robot.lower[active].copy();upper=robot.upper[active].copy()
        if speed_bounded and frame>0:
            lower=np.maximum(lower,previous[active]-robot.speed[active]/fps)
            upper=np.minimum(upper,previous[active]+robot.speed[active]/fps)
        result=least_squares(evaluate,previous[active],jac=lambda x:evaluate(x,True),
                             bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-5)
        previous=q0.copy();previous[active]=result.x
        tf=robot.fk(previous,base)
        fitted.append([tf[t][:3,3] for _,t,_ in MAP]);all_q.append(previous.copy())
        iterations.append(result.nfev);optimizer_success.append(bool(result.success))
    fitted=np.array(fitted);errors=np.linalg.norm(fitted-desired,axis=-1)
    # Ground offset chosen ONCE at frame zero, recorded; no per-frame floor correction.
    tf=robot.fk(all_q[0],bases[0]);floor=min(v[:,2].min() for _,v,_ in robot.vertices(tf))
    bases[:,2,3]-=floor;desired[:,:,2]-=floor;fitted[:,:,2]-=floor;unadjusted_desired[:,:,2]-=floor
    quat=Rotation.from_matrix(bases[:,:3,:3]).as_quat()
    for i in range(1,len(quat)):
        if np.dot(quat[i-1],quat[i])<0:quat[i]*=-1
    np.savez_compressed(destination/'target.npz',q=np.array(all_q),base=bases,base_quat_xyzw=quat,
                        times=np.arange(len(p))/fps,desired=desired,fitted=fitted,errors_m=errors,
                        unadjusted_desired=unadjusted_desired,
                        joint_names=np.array(robot.names),source_contacts=src.get('foot_contacts',np.zeros((len(p),4))),
                        source_positions_world=p,source_names=np.array(names),semantic_heading_offset_rad=np.array(np.pi))
    report={'method':'bounded landmark IK with analytic Jacobian; segment length adaptation; previous-pose and neutral regularization',
            'mapping':[{'source':s,'target':t,'parent_source':pa} for s,t,pa in MAP],
            'source_coordinate_matrix':C.tolist(),'coordinate_determinant':float(np.linalg.det(C)),
            'coordinate_note':'Proper rotation. Native Landau mesh/toes face -Y; C R_source_root R_mount.T (or legacy hip-derived yaw) aligns them with application +Y. Named limbs and root mount are preserved.',
            'semantic_heading_offset_rad':float(np.pi),
            'root_trajectory_scale':float(scale),'constant_floor_shift_m':float(-floor),
            'root_x_mount_preserved':True,'fps':fps,'frame_count':len(p),'duration_s':len(p)/fps,
            'retarget_rmse_m':float(np.sqrt(np.mean(errors**2))),'max_landmark_error_m':float(errors.max()),
            'per_landmark_rmse_m':{t:float(np.sqrt(np.mean(errors[:,i]**2))) for i,(_,t,_) in enumerate(MAP)},
            'anchor_rigid':anchor_rigid,
            'unadjusted_reference_rmse_m':float(np.sqrt(np.mean(np.sum((fitted-unadjusted_desired)**2,axis=-1)))),
            'reference_adjustment_rmse_m':float(np.sqrt(np.mean(np.sum((desired-unadjusted_desired)**2,axis=-1)))),
            'speed_bounded':speed_bounded,'optimizer_nfev':iterations,'optimizer_success':optimizer_success,'max_nfev':max_nfev,
            'orientation_constrained':foot_orientation,'orientation_weights_m':weights,
            'source_rotation_tracking':'foot/toe sole-forward/up plus chest/head forward/up; source root heading' if foot_orientation else 'unavailable: position-only objective; axial twist underdetermined',
            'active_joint_names':[robot.names[i] for i in active],
            'dynamic_feasibility':'not tested','unscaled_source_retained':'source.npz'}
    write_json(destination/'retarget.json',report)
    return report

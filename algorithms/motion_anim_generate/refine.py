"""Bounded CPU retarget alternatives. Original source and raw IK remain inspectable."""
import argparse
import json
import shutil
import subprocess
import numpy as np
import osqp
from scipy import sparse
from landau import Robot
from retarget import MAP, source_foot_contacts
from state import ROOT, OUT, sha256, write_json


def difference_matrices(n):
    return (sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csc'),
            sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc'))


def qp(P,linear,A,lower,upper):
    solver=osqp.OSQP()
    solver.setup(P=sparse.triu(P).tocsc(),q=linear,A=A.tocsc(),l=lower,u=upper,
                 eps_abs=1e-8,eps_rel=1e-8,max_iter=20000,polishing=True,verbose=False)
    result=solver.solve(raise_error=False)
    if result.info.status_val!=1:raise RuntimeError('Temporal QP failed: '+result.info.status)
    return result.x


def project_joints(q,robot,dt,animation_only=False):
    n=len(q);D,D2=difference_matrices(n);I=sparse.eye(n,format='csc')
    A=sparse.vstack([I,D,D2],format='csc');P=I+10*(D2.T@D2)
    result=q.copy()
    for j in robot.active:
        # Leave a numeric margin below validator thresholds, not a changed acceptance gate.
        vmax=robot.speed[j]*.995;amax=76.
        low=np.r_[np.full(n,robot.lower[j]),np.full(n-1,-vmax*dt),np.full(n-2,-amax*dt*dt)]
        high=np.r_[np.full(n,robot.upper[j]),np.full(n-1,vmax*dt),np.full(n-2,amax*dt*dt)]
        if animation_only:
            result[:,j]=qp(P,-q[:,j],I,np.full(n,robot.lower[j]),np.full(n,robot.upper[j]))
        else:
            result[:,j]=qp(P,-q[:,j],A,low,high)
    result[:,robot.locked]=0
    return result


def temporal_contact_refine(run,animation_only=False):
    robot=Robot()
    with np.load(run/'target.npz') as f:data={k:f[k] for k in f.files}
    shutil.copyfile(run/'target.npz',run/'target_raw.npz')
    q=data['q'];base=data['base'].copy();n=len(q);dt=float(np.median(np.diff(data['times'])))
    q=project_joints(q,robot,dt,animation_only=animation_only)
    centers=[];bottom=[];minimum=[]
    for f in range(n):
        verts=robot.vertices(robot.fk(q[f],base[f]));minimum.append(min(v[:,2].min() for _,v,_ in verts))
        c=[];b=[]
        for side in ('l','r'):
            v=np.concatenate([v for link,v,_ in verts if link in ('foot_'+side,'toes_01_'+side)])
            c.append(v.mean(0));b.append(v[:,2].min())
        centers.append(c);bottom.append(b)
    centers=np.array(centers);bottom=np.array(bottom);minimum=np.array(minimum)
    contact=source_foot_contacts(data['source_contacts'])
    # Source stance labels, never invented stance from a desired pass/fail result.
    D,D2=difference_matrices(n);I=sparse.eye(n,format='csc')
    A=sparse.vstack([I,D,D2],format='csc');shifts=[]
    for axis in range(3):
        P=I+20*(D2.T@D2);linear=np.zeros(n)
        low=np.full(n,-.08 if axis<2 else -.06);high=np.full(n,.08 if axis<2 else .06)
        if axis==2:
            low=np.maximum(low,-minimum+.001)
            weights=contact.sum(1)*100.
            target=-np.sum(bottom*contact,axis=1)/np.maximum(contact.sum(1),1)
            P=P+sparse.diags(weights);linear-=weights*target
        else:
            for side in range(2):
                mask=(contact[:-1,side]&contact[1:,side]).astype(float)
                W=sparse.diags(mask*500.)
                P=P+D.T@W@D
                linear+=D.T@W@np.diff(centers[:,side,axis])
        lower=np.r_[low,np.full(n-1,-.015),np.full(n-2,-.003)]
        upper=np.r_[high,np.full(n-1,.015),np.full(n-2,.003)]
        shifts.append(qp(P,linear,A,lower,upper))
    shifts=np.stack(shifts,axis=1);base[:,:3,3]+=shifts
    fitted=[]
    for f in range(n):
        tf=robot.fk(q[f],base[f]);fitted.append([tf[t][:3,3] for _,t,_ in MAP])
    fitted=np.asarray(fitted)
    errors=np.linalg.norm(fitted-data['desired'],axis=-1)
    data.update(q=q,base=base,fitted=fitted,errors_m=errors,base_correction_m=shifts)
    np.savez_compressed(run/'target.npz',**data)
    ret=json.loads((run/'retarget.json').read_text())
    ret['raw_ik_metrics']={k:ret[k] for k in ('retarget_rmse_m','max_landmark_error_m','per_landmark_rmse_m')}
    ret.update(retarget_rmse_m=float(np.sqrt(np.mean(errors**2))),max_landmark_error_m=float(errors.max()),
               unadjusted_reference_rmse_m=float(np.sqrt(np.mean(np.sum((fitted-data['unadjusted_desired'])**2,axis=-1)))),
               per_landmark_rmse_m={t:float(np.sqrt(np.mean(errors[:,i]**2))) for i,(_,t,_) in enumerate(MAP)},
               temporal_contact={'solver':'OSQP 1.0.4','joint_speed_limit_fraction':None if animation_only else .995,'acceleration_bound_rad_s2':None if animation_only else 76,'animation_only':animation_only,
                                 'root_correction_limits_m':[.08,.08,.06],'max_root_correction_m':np.max(np.abs(shifts),axis=0).tolist(),
                                 'rms_joint_change_rad':float(np.sqrt(np.mean((q-np.load(run/'target_raw.npz')['q'])**2))),
                                 'floor_clearance_m':.001,'source_contact_frames':contact.sum(0).tolist(),
                                 'note':'Explicit offline root correction changes source trajectory; error remains measured against pre-correction targets. No physics.'})
    write_json(run/'retarget.json',ret)
    return ret


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent-run',required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--anchor-rigid',action='store_true')
    parser.add_argument('--animation',action='store_true');parser.add_argument('--foot-orientation',action='store_true')
    parser.add_argument('--temporal-contact',action='store_true');args=parser.parse_args()
    for value in [args.parent_run,args.run_id]:
        if not value.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run id')
    parent=OUT/'runs'/args.parent_run;run=OUT/'runs'/args.run_id;run.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(parent/'source.npz',run/'source.npz')
    meta=json.loads((parent/'source_metadata.json').read_text())
    source_node=meta.get('source_node_id',args.parent_run+':generated')
    meta.update(source_node_id=source_node,retarget_parent=args.parent_run,
                retarget_variant={'anchor_rigid':args.anchor_rigid,'temporal_contact':args.temporal_contact,'animation_only':args.animation,'foot_orientation':args.foot_orientation},
                retarget_command=__import__('sys').argv,
                retarget_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                retarget_implementation_sha256={p.name:sha256(p) for p in ROOT.glob('*.py')})
    write_json(run/'source_metadata.json',meta)
    from experiment import process
    result=process(run,'composite',meta['source_kind'],speed_bounded=not args.animation,anchor_rigid=args.anchor_rigid,temporal_contact=args.temporal_contact,animation_only=args.animation,foot_orientation=args.foot_orientation)
    print(json.dumps(result['metrics']))

if __name__=='__main__':main()

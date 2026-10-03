"""Geometric animation diagnostics, using recorded frames and unchanged URDF anatomy."""
import argparse
import json
import numpy as np
from landau import Robot
from retarget import load_source, orientation_axes, source_foot_contacts
from state import OUT, sha256, write_json


def angle(a,b):
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1)))


def diagnose(run):
    src,sk=load_source(run/'source.npz');ix={s:i for i,(s,_) in enumerate(sk)}
    ret=json.loads((run/'retarget.json').read_text());C=np.asarray(ret['source_coordinate_matrix'])
    with np.load(run/'target.npz') as d:q=d['q'];base=d['base']
    r=Robot();rest=r.fk(np.zeros(len(r.names)));axes=orientation_axes(r)
    contacts=source_foot_contacts(src['foot_contacts']);frames=[]
    for f in range(len(q)):
        tf=r.fk(q[f],base[f]);row={'frame':f,'time_s':f/30,'feet':{}}
        for source,target in [('Hips','root_x'),('Chest','spine_03_x'),('Head','head_x')]:
            sf=C@src['global_rot_mats'][f,ix[source],:,2]
            local=rest[target][:3,:3].T@np.array([0.,-1,0])
            forward=tf[target][:3,:3]@local
            row[source+'_forward_dot']=float(forward@sf)
            row[source+'_forward_angle_deg']=float(angle(forward,sf))
        for side,k,prefix in [('l',0,'Left'),('r',1,'Right')]:
            sf=C@src['global_rot_mats'][f,ix[prefix+'Foot'],:,2]
            su=C@src['global_rot_mats'][f,ix[prefix+'Foot'],:,1]
            forward,up=(tf['foot_'+side][:3,:3]@axes['foot_'+side]).T
            yaw_error=np.degrees(np.arctan2(sf[0]*forward[1]-sf[1]*forward[0],sf[:2]@forward[:2]))
            foot_vertices=next(v for link,v,_ in r.vertices(tf) if link=='foot_'+side)
            # Geometric heel/toe mesh ends, not ankle-vs-toe joint height.
            projected=foot_vertices@forward
            heel=foot_vertices[projected<=np.quantile(projected,.25),2].min()
            toe=foot_vertices[projected>=np.quantile(projected,.75),2].min()
            row['feet'][side]={'source_contact':bool(contacts[f,k]),
                'forward_dot':float(forward@sf),'forward_error_deg':float(angle(forward,sf)),
                'sole_normal_error_deg':float(angle(up,su)),
                'sole_tilt_deg':float(angle(up,np.array([0,0,1.]))),
                'source_sole_tilt_deg':float(angle(su,np.array([0,0,1.]))),
                'sole_forward_pitch_deg':float(np.degrees(np.arcsin(np.clip(forward[2],-1,1)))),
                'source_forward_pitch_deg':float(np.degrees(np.arcsin(np.clip(sf[2],-1,1)))),
                'yaw_error_deg':float(yaw_error),'heel_min_z_m':float(heel),'toe_min_z_m':float(toe)}
        frames.append(row)
    summary={part+'_forward_dot_mean':float(np.mean([f[part+'_forward_dot'] for f in frames])) for part in ('Hips','Chest','Head')}
    for side in ('l','r'):
        for metric in ('forward_error_deg','sole_normal_error_deg','sole_tilt_deg'):
            values=[f['feet'][side][metric] for f in frames]
            stance=[f['feet'][side][metric] for f in frames if f['feet'][side]['source_contact']]
            summary[side+'_'+metric+'_mean']=float(np.mean(values))
            summary[side+'_stance_'+metric+'_mean']=float(np.mean(stance)) if stance else None
    report={'purpose':'animation direction and foot-pose diagnostics; no quality rejection cutoff',
            'source_sha256':sha256(run/'source.npz'),'target_sha256':sha256(run/'target.npz'),
            'coordinate_matrix':C.tolist(),'coordinate_determinant':float(np.linalg.det(C)),
            'sole_calibration':'Canonical zero-pose horizontal plane; forward is horizontal ankle-to-toe direction. Actual rest sole planarity audited separately.',
            'summary':summary,'per_frame':frames}
    write_json(run/'directions.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+');args=parser.parse_args()
    for name in args.runs:print(name,json.dumps(diagnose(OUT/'runs'/name)['summary']))

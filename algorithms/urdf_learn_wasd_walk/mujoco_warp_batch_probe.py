"""Real GPU bridge diagnostic: compare observations, contacts and assistance math."""
import argparse
from pathlib import Path
import json
import time

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from algorithms.urdf_learn_wasd_walk.mujoco_backend import OUTPUT, Assistance, digest, write_json
from algorithms.urdf_learn_wasd_walk.mujoco_policy import foot_state, observe, load_actor
from algorithms.urdf_learn_wasd_walk.mujoco_warp_batch import WarpBatch


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--forward-speed',type=float,default=.4)
    args=parser.parse_args();out=OUTPUT/'batch_probes'/args.name
    if not 1<=args.steps<=1000:parser.error('steps must be between1 and1000')
    out.resolve().relative_to(OUTPUT.resolve());out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42);np.random.seed(42);start=time.perf_counter()
    batch=WarpBatch(4,42,0.,stage='forward',gait_reward='load',forward_speed=args.forward_speed)
    policy=None
    if args.checkpoint:
        policy,blob=load_actor(args.checkpoint,batch.model,batch.spec)
        if blob['backend']!='mujoco_warp_cuda' or blob['mujoco_version']!=mujoco.__version__:
            raise ValueError('Diagnostic requires a matching GPU checkpoint')
        policy.to('cuda')
    world=1 if policy is not None else 0  # World0 is the zero-command standing cohort.
    data=mujoco.MjData(batch.model);rows=[]
    for step in range(args.steps):
        # No policy: small reproducible bounded motor excitation only.
        if policy is None:
            action=torch.rand((4,17),device='cuda')*.2-.1
            batch.assistance.coefficient=0. if step<10 else .5
        else:
            with torch.no_grad():action=policy.actor(batch.observations())
            batch.assistance.coefficient=0.
        _,_,done,_=batch.step(action);wp.synchronize()
        mjwarp.get_data_into(data,batch.model,batch.wd,world_id=world)
        gpu_observation=batch.observations()[world,:60].cpu().numpy()
        native_observation=observe(batch.model,data,batch.nominal_q,batch.jids,batch.previous[world].cpu().numpy())
        contacts,slip,clearance,forces=batch.feet(with_forces=True);native_feet=foot_state(batch.model,data,clearance=True)
        native_forces=np.zeros(2);wrench=np.zeros(6)
        for index in range(data.ncon):
            ids=list(data.contact[index].geom)
            if 0 not in ids:continue
            name=batch.model.body(batch.model.geom_bodyid[max(ids)]).name
            side=0 if name in ('foot_l','toes_01_l') else 1 if name in ('foot_r','toes_01_r') else None
            if side is not None:
                mujoco.mj_contactForce(batch.model,data,index,wrench)
                native_forces[side]+=max(float(wrench[0]),0.)
        row={'control_step':step,'observation_max_error':float(np.max(np.abs(gpu_observation-native_observation))),
             'gpu_contacts':contacts[world].tolist(),'native_contacts':[native_feet['left_contact'],native_feet['right_contact']],
             'slip_error_mps':abs(float(slip[world])-native_feet['mean_slip_mps']),
             'normal_force_max_error_n':float(np.max(np.abs(forces[world].cpu().numpy()-native_forces))),
             'clearance_max_error_m':float(np.max(np.abs(clearance[world].cpu().numpy()-np.array([native_feet['left_clearance_m'],native_feet['right_clearance_m']]))))}
        # Recompute the NEXT control wrench from this exact current GPU state.
        batch.assistance_wrench();applied=batch.wrench[world,batch.pelvis].cpu().numpy()
        velocity=np.zeros(6);mujoco.mj_objectVelocity(batch.model,data,mujoco.mjtObj.mjOBJ_BODY,batch.pelvis,velocity,0)
        initial=Rotation.from_quat(batch.reference_quat.cpu().numpy()[[1,2,3,0]]).as_matrix()
        error=Rotation.from_matrix(initial@data.xmat[batch.pelvis].reshape(3,3).T).as_rotvec()
        expected=Assistance(batch.assistance.coefficient).wrench(batch.reference_height-data.xpos[batch.pelvis,2],velocity[5],error,velocity[:3])
        row['wrench_max_error']=float(np.max(np.abs(applied-expected)))
        row.update(raw_mean_actions=action[world].tolist(),
            raw_mean_clip_fraction=float((action[world].abs()>1).float().mean()),
            root_position=data.xpos[batch.pelvis].tolist(),
            action_joint_positions=data.qpos[batch.model.jnt_qposadr[batch.jids]].tolist(),
            target_tracking_error_rad=(data.ctrl[batch.aids]-data.qpos[batch.model.jnt_qposadr[batch.jids]]).tolist(),
            actuator_torques_nm=data.actuator_force[batch.aids].tolist(),
            actuator_effort_bounds=batch.model.actuator_forcerange[batch.aids].tolist(),
            joint_speeds_rad_s=data.qvel[batch.model.jnt_dofadr[batch.jids]].tolist(),
            foot_normal_forces_n=forces[world].tolist(),foot_clearance_m=clearance[world].tolist(),
            training_done_in_any_world=bool(done.any()),selected_world=world,selected_command=float(batch.commands[world]),training_done_in_selected_world=bool(done[world]))
        rows.append(row)
    failures=[]
    for row in rows:
        if row['observation_max_error']>2e-5:failures.append('observation mapping')
        if row['gpu_contacts']!=row['native_contacts']:failures.append('contact classification')
        if row['slip_error_mps']>2e-5:failures.append('contact point velocity')
        if row['normal_force_max_error_n']>2e-5:failures.append('foot normal force')
        if row['clearance_max_error_m']>2e-6:failures.append('mesh clearance')
        if row['wrench_max_error']>2e-5:failures.append('physical assistance mapping')
    result={'passed':not failures,'failures':sorted(set(failures)),'samples':rows,'wall_s':time.perf_counter()-start,
            'seed':42,'milestone_pass':False,'purpose':'GPU bridge correctness; not training or standing proof',
            'source_sha256':digest(__file__),'batch_source_sha256':digest(Path(__file__).with_name('mujoco_warp_batch.py')),
            'auxiliary_assistance_tested':[0.] if policy is not None else [0.,.5],
            'checkpoint_sha256':digest(args.checkpoint) if args.checkpoint else None,
            'action_joint_names':batch.spec['action_joints'],'forward_speed':args.forward_speed,
            'reset_semantics':'Training batch resets on done, explicitly logged; never gate evidence',
            'tensor_device':str(batch.q.device)}
    write_json(out/'result.json',result);print(json.dumps(result),flush=True)
    if failures:raise RuntimeError(result['failures'])


if __name__=='__main__':main()

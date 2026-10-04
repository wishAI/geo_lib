import math

import pytest

from algorithms.urdf_learn_wasd_walk import landau_triangle_navigation as tc
from algorithms.urdf_learn_wasd_walk.landau_direction_contract import MAX_YAW_RATE


def test_ordered_circuit_and_closure():
    tracker=tc.Tracker()
    for i in range(541):tracker.update(tc.point_at(i/10.),i*.1)
    assert tracker.complete
    assert [v['target'] for v in tracker.visits]==[
        'edge_1_midpoint','corner_1','edge_2_midpoint','corner_2','edge_3_midpoint','closure']
    assert tracker.max_corridor<1e-12
    assert tracker.final_distance==0.


def test_return_to_start_does_not_skip_vertices():
    tracker=tc.Tracker()
    for i,point in enumerate([(0.,0.),(0.,9.),(0.,18.),(0.,9.),(0.,0.)]):tracker.update(point,float(i))
    assert not tracker.complete
    assert len(tracker.visits)==2


def test_wrong_way_circuit_cannot_claim_all_corners():
    tracker=tc.Tracker()
    for i in range(541):tracker.update(tc.point_at(54.-i/10.),i*.1)
    assert not tracker.complete


def test_corridor_excursion_is_retained_after_return():
    tracker=tc.Tracker();tracker.update((0.,0.),0.);tracker.update((3.,9.),1.)
    for i in range(541):tracker.update(tc.point_at(i/10.),2.+i*.1)
    assert tracker.complete and tracker.max_corridor==3.


def test_navigation_has_bounded_commands_and_continuous_reference():
    nav=tc.Navigator();reference=previous_yaw=0.
    for i in range(27001):
        t=i*.02;point=tc.point_at(t*.1)
        edge=min(2,int(t*.1/18.));heading=edge*2.*math.pi/3.
        reference+=previous_yaw*.02 if i else 0.
        command=nav.command(point,heading,t)
        assert command[:2]==(.2,0.)
        assert abs(command[2])<=MAX_YAW_RATE+1e-8
        assert nav.reference==pytest.approx(reference,abs=1e-9)
        previous_yaw=command[2]
    assert nav.progress==pytest.approx(54.)


def test_start_does_not_jump_to_closing_edge():
    nav=tc.Navigator();nav.command((0.,0.),0.,0.)
    assert nav.progress==0.
    assert tc.point_at(nav.progress+tc.LOOKAHEAD_M)==(0.,tc.LOOKAHEAD_M)


def test_clock_and_forward_protocol_are_fixed():
    nav=tc.Navigator();nav.command((0.,0.),0.,0.)
    with pytest.raises(ValueError):nav.command((0.,0.),0.,0.)
    with pytest.raises(ValueError):tc.Navigator().command((0.,0.),0.,0.,.1)
    tracker=tc.Tracker();tracker.update((0.,0.),0.)
    with pytest.raises(ValueError):tracker.update((math.nan,0.),1.)


def test_triangle_certificate_rejects_changed_prior_evidence(tmp_path, monkeypatch):
    from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates
    out=tmp_path/'outputs';out.mkdir()
    checkpoint=out/'model_0.pt';checkpoint.write_bytes(b'exact checkpoint')
    evidence=out/'old_dynamics.json';evidence.write_text('original')
    ckpt=dict(path='outputs/model_0.pt',sha256=gates.digest(checkpoint))
    prior=out/'m7.json'
    gates.write(prior,dict(status='passed',milestone='gate_10m_four_directions_no_reset',
        lineage=gates.LINEAGE,backend='mujoco_warp_cuda',checkpoint=ckpt,
        evidence=[dict(kind='forward_dynamics.json',path='outputs/old_dynamics.json',sha256=gates.digest(evidence))],
        cumulative_components=[]))
    milestones=[dict(status='passed') for _ in range(7)]
    milestones[6].update(id='gate_10m_four_directions_no_reset',checkpoint=ckpt,
        evidence=[dict(kind='validation',path='outputs/m7.json',sha256=gates.digest(prior))])
    milestones.append(dict(id=tc.MILESTONE,status='in_progress'))
    ledger=tmp_path/'milestones.json'
    gates.write(ledger,dict(lineage=gates.LINEAGE,backend='mujoco_warp_cuda',milestones=milestones))
    for name,value in [('ROOT',tmp_path),('ALG',tmp_path),('LEDGER',ledger)]:monkeypatch.setattr(gates,name,value)
    evidence.write_text('tampered')
    with pytest.raises(ValueError,match='M7 artifact changed'):
        gates.certify_triangle(out/'certificate',checkpoint,out/'triangle')
    assert not (out/'certificate').exists()
    assert gates.read(ledger)['milestones'][7]['status']=='in_progress'
    # Even an intact certificate cannot transfer success to different weights.
    evidence.write_text('original');checkpoint.write_bytes(b'new checkpoint')
    with pytest.raises(ValueError,match='another checkpoint'):
        gates.certify_triangle(out/'certificate',checkpoint,out/'triangle')


def test_reproduction_retains_prospective_horizon(tmp_path):
    from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates
    checkpoint=tmp_path/'model_0.pt'
    gates.write(tmp_path/'training.json',dict(observation_dim=70))
    result=dict(config=dict(name='triangle',triangle=True,forward=.2,seconds=430.02,triangle_horizon_s=600.))
    job=gates.reproduction_job(result,checkpoint)
    assert '--triangle' in job['args']
    assert job['args'][job['args'].index('--seconds')+1]=='600.0'


def test_triangle_uses_turned_memory_but_straight_gate_cannot(tmp_path):
    from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates
    for name in ('gait_source.py','turn_source.py','controller_source.py'):(tmp_path/name).write_text(name)
    turn_sha=gates.digest(tmp_path/'turn_source.py')
    gates.write(tmp_path/'controller_memory.json',dict(dispatch='command_driven_yaw_extension',
        turn_source_sha256=turn_sha,turned=True,integrated_reference_rad=4.2))
    training=dict(source_sha256={'landau_gait_search.py':gates.digest(tmp_path/'gait_source.py')},
                  command_extension='yaw_v1',turn_source_sha256=turn_sha)
    result=dict(config={'triangle':True},gait_source_sha256=gates.digest(tmp_path/'gait_source.py'),
        controller_source_sha256=gates.digest(tmp_path/'controller_source.py'),turn_source_sha256=turn_sha,
        controller_memory_sha256=gates.digest(tmp_path/'controller_memory.json'))
    gates.check_gait_source(tmp_path,result,training)
    result['config']={}
    with pytest.raises(ValueError,match='Zero-yaw'):
        gates.check_gait_source(tmp_path,result,training)


def test_independent_geometry_checker_replays_commands_and_rejects_tampering(tmp_path):
    """Synthetic kinematic fixture tests geometry/bindings, never a dynamics pass."""
    import numpy as np
    from pathlib import Path
    from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates, mujoco_backend as backend
    from algorithms.urdf_learn_wasd_walk import landau_signed_navigation as signed, landau_direction_contract as dc
    pytest.importorskip('mujoco')
    (tmp_path/'model.xml').write_text('<mujoco><worldbody><body name="base_link"><freejoint/><geom type="sphere" size=".1" mass="1"/><body name="root_x"/></body></worldbody></mujoco>')
    points=[];tracker=tc.Tracker()
    for i in range(541):
        point=tc.point_at(i*.1);points.append(point);tracker.update(point,i*.02)
        if tracker.complete:break
    count=len(points)-1;t=np.arange(count+1)*.02
    q=np.zeros((count+1,7));q[:,:2]=points;q[:,3]=1.
    nav=tc.Navigator();obs=np.zeros((count,70),dtype=np.float32)
    for i in range(count):obs[i,63:66]=nav.command(points[i],0.,float(t[i]))
    np.savez(tmp_path/'policy_trace.npz',observation=obs,time=t[:-1])
    (tmp_path/'turn_source.py').write_text('saved controller fixture')
    turn_sha=gates.digest(tmp_path/'turn_source.py')
    gates.write(tmp_path/'controller_memory.json',dict(simulation_reset=False,memory_version=2,
        turn_source_sha256=turn_sha,integrated_reference_rad=nav.reference))
    (tmp_path/'warp_runtime_source.py').write_bytes((gates.ALG/'mujoco_warp_runtime.py').read_bytes())
    (tmp_path/'warp_health_source.py').write_bytes(Path(backend.__file__).read_bytes())
    metrics=dict(tracker.metrics(),duration_s=float(t[-1]),reset_count=0,done_count=0,fall_count=0,
        max_reference_tilt_rad=0.,root_height_drop_m=0.,max_abs_command=.2,max_abs_action=.5,
        left_foot_liftoff_count=10,right_foot_liftoff_count=10,mean_contact_foot_slip_mps=0.,
        simultaneous_air_fraction=0.,policy_inference_steps=count,control_steps=count,
        leg_joint_excursion_rad={name:.1 for name in ('left_hip_pitch_joint','right_hip_pitch_joint','left_knee_joint','right_knee_joint')})
    runtime_sha=gates.digest(tmp_path/'warp_runtime_source.py')
    result=dict(config=dict(forward=.2,seconds=float(t[-1]),triangle_horizon_s=600.),metrics=metrics,
        identity=dict(warp_runtime_source_sha256=runtime_sha),
        performance=dict(warp_setup=dict(source_sha256=runtime_sha,health_source_sha256=gates.digest(tmp_path/'warp_health_source.py'),
            clock_check='exact float32(previous_raw + float32(dt)); strict increase; logical timestamps count steps')),
        triangle_contract=tc.contract(),policy_trace_sha256=gates.digest(tmp_path/'policy_trace.npz'),triangle_navigation_state=nav.record())
    for name,source in [('triangle_navigation',tc.__file__),('signed_navigation',signed.__file__),('direction_protocol',dc.__file__)]:
        (tmp_path/(name+'_source.py')).write_bytes(Path(source).read_bytes());result[name+'_source_sha256']=gates.digest(source)
    training=dict(command_extension='yaw_v1',memory_version=2,nominal_q=q[0].tolist(),turn_source_sha256=turn_sha)
    gates.check_triangle_trajectory(tmp_path,result,training,q,t,count)
    obs[200,65]+=.001
    np.savez(tmp_path/'policy_trace.npz',observation=obs,time=t[:-1]);result['policy_trace_sha256']=gates.digest(tmp_path/'policy_trace.npz')
    with pytest.raises(ValueError,match='commands differ'):
        gates.check_triangle_trajectory(tmp_path,result,training,q,t,count)


def test_candidate_prerequisites_preserve_history_and_are_rechecked(tmp_path,monkeypatch):
    """Certificate orchestration fixture; physical validators are tested separately."""
    import copy
    from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates
    out=tmp_path/'outputs';out.mkdir();checkpoint=out/'model_0.pt';checkpoint.write_bytes(b'new candidate')
    gates.write(out/'training.json',dict(policy_family='fixture'))
    milestones=[dict(id='m'+str(i),status='passed') for i in range(12)]
    milestones[6].update(id='gate_10m_four_directions_no_reset',checkpoint={'sha256':'historical'},evidence=[dict(kind='validation',path='unused',sha256='historical')])
    milestones[7].update(id=tc.MILESTONE,status='in_progress');milestones[8]['status']='not_started'
    ledger=tmp_path/'milestones.json'
    original=dict(lineage=gates.LINEAGE,backend='mujoco_warp_cuda',milestones=milestones,sessionStatus={})
    gates.write(ledger,original)
    for key,value in [('ROOT',tmp_path),('ALG',tmp_path),('LEDGER',ledger)]:monkeypatch.setattr(gates,key,value)
    calls=[]
    def checked(folder,*args,**kwargs):
        calls.append((folder.name,kwargs))
        return dict(created_at='2099-01-01T00:00:00+00:00',config=dict(direction_navigation='hysteretic_v6')),{'fixture':True}
    monkeypatch.setattr(gates,'check_standing',checked);monkeypatch.setattr(gates,'check_walking',checked)
    monkeypatch.setattr(gates,'reproduction_job',lambda *a:{'fixture':True})
    gui=[];monkeypatch.setattr(gates,'sync_gui',lambda *a:gui.append(a))
    names=('dynamics.json','proof_metadata.json','visual_review.json','proof.mp4','trajectory.npz',
        'policy_trace.npz','foot_trace.json','direction_protocol_source.py','controller_memory.json',
        'turn_source.py','gait_source.py','controller_source.py','backend_source.py','model.xml',
        'signed_navigation_source.py','triangle_navigation_source.py','residual_source.py',
        'active_residual_source.py','renderer_source.py','warp_runtime_source.py','warp_health_source.py')
    directions=[];cumulative=[]
    for name in ('forward','left','right','backward',*[f'm{i}' for i in range(1,7)],'triangle'):
        folder=out/name;folder.mkdir()
        for filename in names:(folder/filename).write_text('fixture '+filename)
        gates.write(folder/'dynamics.json',dict(config={'direction':name}))
        if name in ('forward','left','right','backward'):directions.append(folder)
        elif name!='triangle':cumulative.append(folder)
    certificate=out/'prerequisites'
    final=gates.certify_directions(certificate,checkpoint,directions,cumulative,promote=False)
    assert final['canonical_milestone_pass'] is False and len(calls)==10
    assert gates.read(ledger)==original and gui==[]
    path=certificate/'milestone_validation.json';sha=gates.digest(path)
    with pytest.raises(ValueError,match='SHA256 required'):
        gates.certify_triangle(out/'m8cert',checkpoint,out/'triangle',prerequisite_certificate=path)
    wrong=copy.deepcopy(final);wrong['checkpoint']['sha256']='0'*64;gates.write(path,wrong)
    with pytest.raises(ValueError,match='another checkpoint'):
        gates.certify_triangle(out/'m8cert',checkpoint,out/'triangle',prerequisite_certificate=path,prerequisite_sha256=gates.digest(path))
    gates.write(path,final)
    assert gates.digest(path)==sha
    evidence=directions[0]/'foot_trace.json';saved=evidence.read_bytes();evidence.write_text('tampered')
    with pytest.raises(ValueError,match='M7 artifact changed'):
        gates.certify_triangle(out/'m8cert',checkpoint,out/'triangle',prerequisite_certificate=path,prerequisite_sha256=sha)
    evidence.write_bytes(saved);calls.clear()
    gates.certify_triangle(out/'m8cert',checkpoint,out/'triangle',prerequisite_certificate=path,prerequisite_sha256=sha)
    assert len(calls)==11
    updated=gates.read(ledger)
    assert updated['milestones'][:7]==original['milestones'][:7]
    assert updated['milestones'][7]['status']=='passed' and updated['milestones'][8]['status']=='in_progress'

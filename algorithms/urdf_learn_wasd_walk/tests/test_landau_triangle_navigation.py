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

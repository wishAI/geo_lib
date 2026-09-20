import json
import numpy as np
import pytest
from hand_refine import anatomical_basis, calibration, refine
from landau import Robot
from retarget import MAP, C, load_source
from state import OUT, write_json


def test_anatomical_frame_is_proper_on_both_sides():
    for sign in (-1,1):
        basis=anatomical_basis([sign,0,0],[sign,.2,.4])
        assert np.allclose(basis.T@basis,np.eye(3))
        assert np.linalg.det(basis)==pytest.approx(1)
        assert np.allclose(basis[:,0],[sign,0,0])
    for finger,thumb in [([0,0,0],[1,0,0]),([1,0,0],[2,0,0]),([np.nan,0,0],[0,1,0])]:
        with pytest.raises(ValueError):anatomical_basis(finger,thumb)


def test_known_wrist_rotation_recovered_without_moving_landmarks(tmp_path):
    r=Robot();cal=calibration(r);base=np.eye(4);q=np.zeros(len(r.names));wanted=q.copy()
    active=[i for i,n in enumerate(r.names) if 'forearm_roll' in n or 'wrist_pitch' in n]
    q[active]=1e-15  # Regression: tiny nonzero TRF radius previously yielded no correction.
    wanted[active]=[.5,.4,-.3,.2]
    tf=r.fk(q);goal=r.fk(wanted)
    example=OUT/'vendor/kimodo/kimodo/assets/demo/examples/kimodo-soma-rp/02_multi_text_prompt/motion.npz'
    source,sk=load_source(example);source={k:v[:3].copy() for k,v in source.items()};ix={n:i for i,(n,_) in enumerate(sk)}
    for side,prefix in [('l','Left'),('r','Right')]:
        source['global_rot_mats'][:,ix[prefix+'Hand']]=C.T@goal['hand_'+side][:3,:3]@cal[side]['target_hand_local_basis']@cal[side]['source_neutral_basis'].T
    parent=tmp_path/'parent';run=tmp_path/'run';parent.mkdir();run.mkdir()
    np.savez(run/'source.npz',**source)
    np.savez(parent/'target.npz',q=np.tile(q,(3,1)),base=np.tile(base,(3,1,1)),
        fitted=np.tile(np.array([tf[t][:3,3] for _,t,_ in MAP]),(3,1,1)),times=np.arange(3)/30)
    write_json(parent/'retarget.json',{'source_coordinate_matrix':C.tolist()})
    report=refine(parent,run)
    for side in ('l','r'):
        assert report['statistics'][side+'_finger_before_error_deg']['mean']>5
        assert report['statistics'][side+'_finger_after_error_deg']['mean']<3
    assert report['max_landmark_position_change_m']<1e-8
    with np.load(run/'target.npz') as data:
        other=[i for i in range(len(q)) if i not in active]
        assert np.array_equal(data['q'][:,other],np.tile(q[other],(3,1)))


def test_progress_does_not_relabel_stale_run_metrics(tmp_path,monkeypatch):
    import state
    monkeypatch.setattr(state,'OUT',tmp_path)
    state.progress('a',active_run='seed44',metrics={'rmse':.03},config={'seed':44},artifacts=['old'])
    state.progress('b',active_run='seed42')
    report=json.loads((tmp_path/'backend_progress.json').read_text())
    assert report['active_run']=='seed42'
    assert all(k not in report for k in ['metrics','config','artifacts'])

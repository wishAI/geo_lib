import numpy as np
from landau import Robot
from retarget import C, skeleton_names


def test_canonical_assets_and_partition():
    r=Robot(); a=r.audit()
    assert a['mesh_count']==68 and a['triangle_count']==8864
    assert len(r.moving)==69 and len(r.links)==71
    assert set(r.active).isdisjoint(r.locked)
    assert len(r.active)+len(r.locked)==69
    assert 'left_elbow_joint' in a['action_joints']
    assert all('shin_roll' in r.names[i] or any(f in r.names[i] for f in ('thumb','index','middle','ring','pinky')) for i in r.locked)


def test_root_mount_and_y_forward_transport():
    r=Robot();q=np.zeros(len(r.names));rest=r.fk(q)
    assert np.allclose(rest['root_x'][:3,:3] @ [0,1,0], [0,0,1],atol=1e-5)
    base=np.eye(4);base[1,3]=.2
    moved=r.fk(q,base)
    assert np.allclose(moved['foot_l'][:3,3]-rest['foot_l'][:3,3],[0,.2,0])
    assert rest['thigh_stretch_l'][0,3]>rest['thigh_stretch_r'][0,3]


def test_semantic_coordinate_conversion_and_skeleton():
    assert np.allclose(C @ [0,0,1],[0,1,0])
    assert np.allclose(C @ [0,1,0],[0,0,1])
    assert np.linalg.det(C)==-1
    assert len(skeleton_names(30))==30
    assert len(skeleton_names(77))==77


def test_bounded_ik_respects_speed_and_locks(tmp_path):
    from retarget import solve, ROOT
    src=ROOT/'outputs/vendor/kimodo/kimodo/assets/demo/examples/kimodo-soma-rp/02_multi_text_prompt/motion.npz'
    with np.load(src) as f:
        np.savez(tmp_path/'source.npz',**{k:f[k][:8] for k in f.files})
    solve(tmp_path/'source.npz',tmp_path,max_nfev=12,speed_bounded=True)
    r=Robot()
    with np.load(tmp_path/'target.npz') as d:
        assert np.all(np.abs(np.diff(d['q'],axis=0))*30 <= r.speed+1e-6)
        assert np.all(d['q'][:,r.locked]==0)
        assert np.all(d['q']>=r.lower-1e-9) and np.all(d['q']<=r.upper+1e-9)

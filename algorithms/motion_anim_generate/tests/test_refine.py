import numpy as np
from landau import Robot
from refine import project_joints
from retarget import solve, ROOT, MAP


def test_qp_preserves_static_and_bounds_reversal():
    r=Robot();q=np.zeros((31,len(r.names)));j=r.active[0]
    stationary=project_joints(q,r,1/30)
    assert np.max(np.abs(stationary))<1e-8
    q[8:17,j]=.65;q[17:24,j]=-.5
    fitted=project_joints(q,r,1/30)
    assert np.max(np.abs(np.diff(fitted,axis=0))*30)<=4+1e-6
    assert np.max(np.abs(np.diff(fitted,n=2,axis=0))*900)<=76+1e-4
    assert np.all(fitted>=r.lower-1e-7) and np.all(fitted<=r.upper+1e-7)
    assert np.all(fitted[:,r.locked]==0)


def test_rigid_anchor_targets_are_reachable_and_old_reference_retained(tmp_path):
    src=ROOT/'outputs/vendor/kimodo/kimodo/assets/demo/examples/kimodo-soma-rp/02_multi_text_prompt/motion.npz'
    with np.load(src) as f:np.savez(tmp_path/'source.npz',**{k:f[k][:8] for k in f.files})
    solve(tmp_path/'source.npz',tmp_path,max_nfev=12,speed_bounded=True,anchor_rigid=True)
    with np.load(tmp_path/'target.npz') as data:
        for target in ['spine_01_x','thigh_stretch_l','thigh_stretch_r']:
            i=[t for _,t,_ in MAP].index(target)
            assert np.max(np.abs(data['desired'][:,i]-data['fitted'][:,i]))<1e-10
        assert np.max(np.abs(data['desired']-data['unadjusted_desired']))>.02


def test_source_contact_30_and_77_channel_order():
    from retarget import source_foot_contacts
    # Toe-end L contact must never be mislabeled as right-foot contact.
    assert np.array_equal(source_foot_contacts([[0,0,1,0,0,0]]),[[True,False]])
    assert np.array_equal(source_foot_contacts([[0,0,0,1,0,0]]),[[False,True]])
    assert np.array_equal(source_foot_contacts([[0,1,0,0]]),[[True,False]])


def test_orientation_retarget_tracks_root_and_sole(tmp_path):
    from retarget import C, load_source, orientation_axes
    src=ROOT/'outputs/vendor/kimodo/kimodo/assets/demo/examples/kimodo-soma-rp/02_multi_text_prompt/motion.npz'
    with np.load(src) as f:np.savez(tmp_path/'source.npz',**{k:f[k][:8] for k in f.files})
    solve(tmp_path/'source.npz',tmp_path,anchor_rigid=True,foot_orientation=True)
    source,skeleton=load_source(tmp_path/'source.npz');ix={s:i for i,(s,_) in enumerate(skeleton)}
    r=Robot();axes=orientation_axes(r)
    with np.load(tmp_path/'target.npz') as d:
        for frame in range(8):
            tf=r.fk(d['q'][frame],d['base'][frame])
            assert np.allclose(tf['root_x'][:3,:3],C@source['global_rot_mats'][frame,ix['Hips']],atol=1e-6)
            for side,prefix in [('l','Left'),('r','Right')]:
                up=tf['foot_'+side][:3,:3]@axes['foot_'+side][:,1]
                desired=C@source['global_rot_mats'][frame,ix[prefix+'Foot'],:,1]
                assert up@desired>.98

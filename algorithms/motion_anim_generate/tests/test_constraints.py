"""Exercise the pinned upstream CFG branch, including the misleading regular-zero case."""
import importlib.util
import pytest
import numpy as np
from state import OUT


def test_root_path_measures_smooth_feature_not_hips_or_vertical_axis():
    from constraint_smoke import PATH_FRAMES, PATH_XZ, root_path_metrics
    frames=np.arange(60)
    z=np.interp(frames,PATH_FRAMES,PATH_XZ[:,1])
    smooth=np.stack([np.zeros(60),np.ones(60),z],axis=-1)
    sample={'smooth_root_pos':smooth,'root_positions':smooth+np.array([.02,0,0]),
            'global_root_heading':np.tile([1.,0.],(60,1))}
    result=root_path_metrics(sample)
    assert result['waypoint_rms_m']==0
    assert result['smooth_root_final_displacement_xz_m']==[0.,.4]
    # A mistaken Y/Z interpretation must not receive a perfect score.
    sample['smooth_root_pos']=smooth[:,[0,2,1]]
    assert root_path_metrics(sample)['waypoint_rms_m']>.7
    sample['smooth_root_pos'][4,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):root_path_metrics(sample)


def test_official_root_path_constructor_uses_xz_pairs(tmp_path):
    import sys
    vendor=OUT/'vendor/kimodo'
    if not vendor.exists():pytest.skip('Pinned ignored vendor unavailable')
    sys.path.insert(0,str(vendor))
    from constraint_smoke import make_root_path, PATH_FRAMES, PATH_XZ
    constraint=make_root_path(None,'cpu',tmp_path)
    data={'smooth_root_2d':[]};indices={'smooth_root_2d':[]}
    constraint.update_constraints(data,indices)
    np.testing.assert_allclose(data['smooth_root_2d'][0].numpy(),PATH_XZ)
    np.testing.assert_array_equal(indices['smooth_root_2d'][0].numpy(),PATH_FRAMES)
    assert constraint.global_root_heading is None
    from kimodo.skeleton import SOMASkeleton30
    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    rep=KimodoMotionRep(SOMASkeleton30(),30)
    observed,mask=rep.create_conditions_from_constraints([constraint],60,False,'cpu')
    root=observed[:,rep.slice_dict['smooth_root_pos']]
    root_mask=mask[:,rep.slice_dict['smooth_root_pos']]
    np.testing.assert_allclose(root.numpy()[PATH_FRAMES][:,[0,2]],PATH_XZ)
    assert int(mask.sum())==10  # only XZ at the five supplied frames
    assert not root_mask[:,1].any()  # no accidental vertical conditioning


def test_official_constraint_branch_survives_without_text():
    torch = pytest.importorskip('torch')
    path = OUT/'vendor/kimodo/kimodo/model/cfg.py'
    if not path.exists():
        pytest.skip('Pinned vendor is a task-local ignored dependency')
    spec = importlib.util.spec_from_file_location('tested_kimodo_cfg', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Denoiser(torch.nn.Module):
        def forward(self, x, x_pad_mask, text_feat, text_feat_pad_mask, timesteps,
                    first_heading_angle=None, motion_mask=None, observed_motion=None):
            evidence=(motion_mask*observed_motion).reshape(len(x),-1).sum(1)
            return torch.ones_like(x)*evidence[:,None,None]

    wrapper=module.ClassifierFreeGuidedModel(Denoiser())
    inputs=dict(x=torch.zeros(1,2,1),x_pad_mask=torch.ones(1,2,dtype=torch.bool),
        text_feat=torch.zeros(1,1,4),text_feat_pad_mask=torch.ones(1,1,dtype=torch.bool),
        timesteps=torch.zeros(1),motion_mask=torch.ones(1,2,1),observed_motion=torch.ones(1,2,1))
    null=wrapper(cfg_type='regular',cfg_weight=0.,**inputs)
    separated=wrapper(cfg_type='separated',cfg_weight=[0.,2.],**inputs)
    assert torch.equal(null,torch.zeros_like(null))
    assert torch.equal(separated,torch.full_like(separated,4.))
    inputs['motion_mask'].zero_()
    assert torch.equal(wrapper(cfg_type='separated',cfg_weight=[0.,2.],**inputs),null)

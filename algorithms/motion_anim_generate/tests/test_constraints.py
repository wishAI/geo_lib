"""Exercise the pinned upstream CFG branch, including the misleading regular-zero case."""
import importlib.util
import pytest
from state import OUT


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

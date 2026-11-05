import copy
import json
from unittest.mock import patch

import pytest
import torch

from algorithms.urdf_learn_wasd_walk import landau_hand_transfer as transfer
from algorithms.urdf_learn_wasd_walk.mujoco_milestones import digest


@pytest.fixture
def frozen_transfer(tmp_path):
    donor, target = tmp_path/'donor', tmp_path/'target'
    donor.mkdir(); target.mkdir()
    original = {'parameters':torch.arange(16), 'standing_actor_state_dict':{'weight':torch.ones(3)}}
    torch.save(original, donor/'model_0.pt')
    for name in ('model.xml','control_source.py'):
        (donor/name).write_text(name)
        (target/name).write_text(name)
    (target/'hand_transfer_source.py').write_bytes(transfer.Path(transfer.__file__).read_bytes())
    old = dict(nominal_q=[0.,0.],nominal_ctrl=[0.],reference_xy=[0.,0.],
               arguments={'name':'donor','forward':.2},status='completed',
               checkpoints={'model_0.pt':digest(donor/'model_0.pt')},
               canonical_milestone_pass=False,created_at='before',policy_composition='donor')
    (donor/'training.json').write_text(json.dumps(old))
    record = dict(version=transfer.VERSION,checkpoint=str(donor/'model_0.pt'),
                  checkpoint_sha256=digest(donor/'model_0.pt'),training_sha256=digest(donor/'training.json'),
                  source_sha256=digest(transfer.__file__),files={n:digest(donor/n) for n in ('model.xml','control_source.py')})
    meta = dict(old,hand_pose='closed_fists_v1',hand_pose_transfer=record,
                nominal_q=[0.,1.],nominal_ctrl=[1.],status='transferred_requires_cumulative_validation')
    blob = dict(original,hand_pose_transfer=record)
    with patch.object(transfer,'transferred_controls',return_value=([0.,1.],[1.])):
        yield meta,blob,target


def test_frozen_transfer_is_explicit_and_not_certified(frozen_transfer):
    meta,blob,folder=frozen_transfer
    _,old,original=transfer.check_contract(meta,blob,folder,digest)
    assert old['nominal_q'] != meta['nominal_q']
    assert torch.equal(original['parameters'],blob['parameters'])


@pytest.mark.parametrize('change', ['network','nonfinger','command','extra','donor','source','credit'])
def test_transfer_rejects_undeclared_changes(frozen_transfer,change):
    meta,blob,folder=copy.deepcopy(frozen_transfer)
    if change=='network':blob['standing_actor_state_dict']['weight'][0]=2
    elif change=='nonfinger':meta['nominal_q'][0]=.1
    elif change=='command':meta['arguments']['forward']=.4
    elif change=='extra':meta['unknown_override']=True
    elif change=='donor':meta['hand_pose_transfer']['checkpoint_sha256']='wrong'
    elif change=='source':(folder/'control_source.py').write_text('changed')
    elif change=='credit':meta['canonical_milestone_pass']=True
    with pytest.raises(ValueError):transfer.check_contract(meta,blob,folder,digest)

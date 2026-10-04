"""Compose right-only sway calibration with two unchanged learned corrections."""
import argparse
import ast
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

VARIANT='right_balance_rebased_dual_residual_v1'
FROZEN_INDICES=[i for i in range(20) if i not in (12,13)]
RESIDUAL_KEYS=('residual_source_sha256','active_residual_source_sha256',
    'residual_bound_rad','residual_feature_dim','residual_action_joints','residual_gate',
    'active_residual_bound_rad','active_residual_feature_dim','active_residual_action_joints','active_residual_gate')


def same_weights(left,right):
    import torch
    return left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)


def deployment_ast(path):
    """Compare deployment at the required CEM20 arity, excluding dead21+ code."""
    names={'CommandMemory','commanded_action','configure_profile','command_profile','evaluate_turn_gate'}
    tree=ast.parse(Path(path).read_text())
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes}!=names:raise ValueError('Incomplete saved turn implementation')
    constants=[node for node in tree.body if isinstance(node,ast.Assign) and
               any(isinstance(target,ast.Name) and target.id in {'TURN_START','TURN_END','HOLD_START','TURN_RATE'} for target in node.targets)]
    if len(constants)!=4:raise ValueError('Incomplete saved turn constants')
    class FixedArity(ast.NodeTransformer):
        def visit_If(self,node):
            test=node.test
            # Later sources also support21/22parameters; these branches cannot
            # execute with the exact20parameter tensors required by this variant.
            first=test.values[0] if isinstance(test,ast.BoolOp) and isinstance(test.op,ast.And) else test
            if (isinstance(first,ast.Compare) and ast.dump(first.left)==ast.dump(ast.parse('p.shape[1]',mode='eval').body)
                and len(first.ops)==1 and isinstance(first.ops[0],ast.GtE) and
                isinstance(first.comparators[0],ast.Constant) and isinstance(first.comparators[0].value,int) and first.comparators[0].value>20):
                return [self.visit(n) for n in node.orelse]
            return self.generic_visit(node)
    return ast.dump(FixedArity().visit(ast.Module(body=constants+nodes,type_ignores=[])),include_attributes=False)


def check_contract(meta,blob,digest):
    import torch
    from algorithms.urdf_learn_wasd_walk.landau_cruise_adaptation import donor
    if meta.get('residual_variant')!=VARIANT or blob.get('residual_variant')!=VARIANT:
        raise ValueError('Right rebase variant mismatch')
    if meta.get('policy_family')!='periodic_feedback_residual_ppo' or any(k.startswith(('frozen_cruise_','frozen_active_','cruise_initialization_')) for k in meta):
        raise ValueError('Right rebase has misleading frozen lineage')
    path=Path(meta['frozen_dual_checkpoint'])
    if digest(path)!=meta['frozen_dual_checkpoint_sha256'] or digest(path.parent/'training.json')!=meta['frozen_dual_training_sha256']:
        raise ValueError('Frozen dual input changed')
    old,original=donor(path,digest)
    for key in RESIDUAL_KEYS:
        if meta.get(key)!=old[key]:raise ValueError('Frozen dual contract changed: '+key)
    if tuple(blob['turn_parameters'].shape)!=(20,) or not torch.equal(blob['turn_parameters'][FROZEN_INDICES],original['turn_parameters'][FROZEN_INDICES]):
        raise ValueError('Only right sway parameters12/13 may change')
    p=blob['turn_parameters']
    if not bool(torch.isfinite(p).all()) or not (-.06<=float(p[12])<=.03 and -.25<=float(p[13])<=.25):
        raise ValueError('Right sway outside bounded calibration')
    if not torch.equal(blob['parameters'],original['parameters']):raise ValueError('Frozen gait changed')
    for key in ('standing_actor_state_dict','residual_actor_state_dict','active_residual_actor_state_dict'):
        if not same_weights(blob[key],original[key]):raise ValueError('Frozen dual weights changed: '+key)
    parent=Path(meta['parent_checkpoint']);pm=json.loads((parent.parent/'training.json').read_text())
    if (digest(parent)!=meta['parent_checkpoint_sha256'] or digest(parent.parent/'training.json')!=meta['parent_training_sha256'] or
        pm.get('policy_family')!='periodic_feedback_cem' or str(pm.get('status','')).startswith('invalidated_') or pm['checkpoints'][parent.name]!=digest(parent)):
        raise ValueError('Right calibration parent changed')
    duration=meta.get('arguments',{}).get('turn_duration')
    if not isinstance(duration,(int,float)) or not 14.<=duration<=90. or duration!=old.get('arguments',{}).get('turn_duration') or duration!=pm.get('arguments',{}).get('turn_duration'):
        raise ValueError('Frozen turn duration changed')
    fresh=torch.load(parent,map_location='cpu',weights_only=False)
    if any(not torch.equal(blob[k],fresh[k]) for k in ('parameters','turn_parameters')) or not same_weights(blob['standing_actor_state_dict'],fresh['standing_actor_state_dict']):
        raise ValueError('Right calibration tensors changed')
    if meta['turn_source_sha256']!=pm['turn_source_sha256'] or digest(parent.parent/'turn_source.py')!=meta['turn_source_sha256']:
        raise ValueError('Right calibration source changed')
    if deployment_ast(parent.parent/'turn_source.py')!=deployment_ast(path.parent/'turn_source.py'):
        raise ValueError('Right rebase must preserve deployment code')
    return True


def compose(parent,previous,name):
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    from algorithms.urdf_learn_wasd_walk.landau_cruise_adaptation import donor
    parent=Path(parent).resolve();previous=Path(previous).resolve()
    for path in (parent,previous):path.relative_to((backend.OUTPUT/'training').resolve())
    old,original=donor(previous,backend.digest)
    pm=json.loads((parent.parent/'training.json').read_text())
    fresh=torch.load(parent,map_location='cpu',weights_only=False)
    for key in ('model_xml_sha256','urdf_sha256','mesh_tree_sha256','action_joints','action_scale','nominal_q','nominal_ctrl'):
        if old[key]!=pm[key]:raise ValueError('Right composition identity mismatch: '+key)
    if backend.digest(parent.parent/'control_source.py')!=backend.digest(previous.parent/'control_source.py'):
        raise ValueError('Frozen gait source changed')
    if old['arguments']['turn_duration']!=pm['arguments']['turn_duration']:
        raise ValueError('Turn command duration changed')
    blob=copy.deepcopy(original);blob['turn_parameters']=fresh['turn_parameters'].clone();blob['residual_variant']=VARIANT
    meta={k:copy.deepcopy(v) for k,v in old.items() if not k.startswith('frozen_cruise_')}
    meta.pop('composition',None)
    meta.update(residual_variant=VARIANT,parent_checkpoint=str(parent),parent_checkpoint_sha256=backend.digest(parent),
        parent_training_sha256=backend.digest(parent.parent/'training.json'),turn_source_sha256=pm['turn_source_sha256'],
        source_sha256=pm['source_sha256'],frozen_dual_checkpoint=str(previous),frozen_dual_checkpoint_sha256=backend.digest(previous),
        frozen_dual_training_sha256=backend.digest(previous.parent/'training.json'),created_at=datetime.now(timezone.utc).isoformat(),
        status='composed_requires_nominal_validation',canonical_milestone_pass=False,
        policy_composition='Right sway12/13 calibration; unchanged cruise/active residuals, all other turn parameters, gait and standing',
        right_rebase={'changed_turn_indices':[i for i in (12,13) if original['turn_parameters'][i]!=fresh['turn_parameters'][i]],
            'source_sha256':backend.digest(__file__),'new_ppo_training':False})
    meta['arguments']={**old['arguments'],'name':name}
    check_contract(meta,blob,backend.digest)
    folder=(backend.OUTPUT/'training'/name).resolve();folder.relative_to((backend.OUTPUT/'training').resolve());folder.mkdir(exist_ok=False)
    for filename in ('control_source.py','residual_source.py','active_residual_source.py'):
        (folder/filename).write_bytes((previous.parent/filename).read_bytes())
    for filename in ('turn_source.py','teleop_contract.py','model.xml'):
        (folder/filename).write_bytes((parent.parent/filename).read_bytes())
    (folder/'right_rebase_source.py').write_bytes(Path(__file__).read_bytes())
    torch.save(blob,folder/'model_0.pt');meta['checkpoints']={'model_0.pt':backend.digest(folder/'model_0.pt')}
    backend.write_json(folder/'training.json',meta)
    return folder/'model_0.pt'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--frozen-dual-checkpoint',required=True)
    parser.add_argument('--name',required=True);args=parser.parse_args()
    print(compose(args.checkpoint,args.frozen_dual_checkpoint,args.name))


if __name__=='__main__':main()

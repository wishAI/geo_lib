"""PPO probability regression: command-conditioned exploration is minibatch-safe."""
import unittest
try:
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
except ImportError:
    torch = None
from algorithms.urdf_learn_wasd_walk.landau_forward_control import make_actor, install_moving_exploration_floor, initialize_fresh_moving, policy_mean


@unittest.skipIf(torch is None, 'Requires the pinned training environment')
class ExplorationTests(unittest.TestCase):
    def test_turn_extension_preserves_exact_uncommanded_branch(self):
        from types import SimpleNamespace
        from algorithms.urdf_learn_wasd_walk import landau_gait_search as base
        from algorithms.urdf_learn_wasd_walk.landau_turn_control import commanded_action
        names=[side+'_'+joint+'_joint' for side in ('left','right') for joint in ('hip_pitch','hip_yaw','hip_roll','knee','ankle_pitch','toe')]+['waist_yaw_joint','waist_roll_joint','waist_pitch_joint','left_shoulder_pitch_joint','right_shoulder_pitch_joint']
        prior=make_actor(63);prior.eval()
        obs=torch.zeros(2,70);obs[:,8]=-1.;obs[:,62]=.5;obs[1,63]=.2
        walking=torch.tensor([(lo+hi)/2 for lo,hi in base.PARAMETERS.values()])
        with torch.no_grad():
            expected=base.evaluate_mean(SimpleNamespace(parameters=walking,standing_prior=prior),obs,names)
            actual=commanded_action(base,walking,prior,obs,obs[:,:63],torch.tensor([.3,-.3,0.,-.04]),names,0.,False)
        self.assertTrue(torch.equal(actual,expected))

    def test_restart_parameters_preserve_turn_when_unused_or_neutral(self):
        from algorithms.urdf_learn_wasd_walk import landau_gait_search as base
        from algorithms.urdf_learn_wasd_walk.landau_turn_control import commanded_action,CommandMemory
        names=[side+'_'+joint+'_joint' for side in ('left','right') for joint in ('hip_pitch','hip_yaw','hip_roll','knee','ankle_pitch','toe')]+['waist_yaw_joint','waist_roll_joint','waist_pitch_joint','left_shoulder_pitch_joint','right_shoulder_pitch_joint']
        prior=make_actor(63);prior.eval()
        obs=torch.zeros(2,70);obs[:,8]=-1.;obs[:,62]=.1;obs[:,63]=.2;obs[:,65]=.0357
        walking=torch.tensor([(lo+hi)/2 for lo,hi in base.PARAMETERS.values()])
        turn=torch.tensor([.1,-.3,.04,.01,-.01,-.01,-.02])
        extended=torch.cat((turn,torch.tensor([1.,.04,0.,0.,.07])))
        memory=CommandMemory();memory.turned=True;memory.previous_time=27.
        with torch.no_grad():
            expected=commanded_action(base,walking,prior,obs,obs[:,:63],turn,names,.4,True)
            unused=extended.clone();unused[9:11]=torch.tensor([2.,2.])
            actual=commanded_action(base,walking,prior,obs,obs[:,:63],unused,names,.4,True,memory)
            self.assertTrue(torch.equal(actual,expected))
            memory.restart_time=25.
            actual=commanded_action(base,walking,prior,obs,obs[:,:63],extended,names,.4,True,memory)
            self.assertTrue(torch.equal(actual,expected))

    def test_saved_parametric_source_is_hash_bound(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from algorithms.urdf_learn_wasd_walk.landau_forward_control import load_gait_source
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'control_source.py';path.write_text('version = 123\n')
            metadata={'source_sha256':{'/recorded/landau_gait_search.py':hashlib.sha256(path.read_bytes()).hexdigest()}}
            module,_=load_gait_source(Path(directory)/'model_0.pt',metadata)
            self.assertEqual(module.version,123)
            path.write_text('version = 456\n')
            with self.assertRaisesRegex(ValueError,'source changed'):
                load_gait_source(Path(directory)/'model_0.pt',metadata)

    def test_parametric_gait_uses_feedback_and_preserves_saved_standing(self):
        from types import SimpleNamespace
        from algorithms.urdf_learn_wasd_walk.landau_gait_search import PARAMETERS, gait_action, evaluate_mean
        names=[side+'_'+joint+'_joint' for side in ('left','right') for joint in ('hip_pitch','hip_yaw','hip_roll','knee','ankle_pitch','toe')]+['waist_yaw_joint','waist_roll_joint','waist_pitch_joint','left_shoulder_pitch_joint','right_shoulder_pitch_joint']
        prior=make_actor(63);prior.eval()
        raw=torch.zeros(2,70);raw[:,8]=-1.;raw[:,62]=1.-1.8/30.;raw[:,63]=.2
        parameters=torch.tensor([(lo+hi)/2 for lo,hi in PARAMETERS.values()]);parameters[5]=.8
        policy=SimpleNamespace(parameters=parameters,standing_prior=prior)
        with torch.no_grad():
            neutral=gait_action(raw,parameters,names)
            tilted=raw.clone();tilted[:,7]=-.1
            self.assertGreater(float((gait_action(tilted,parameters,names)-neutral).abs().max()),.1)
            raw[0,63]=0.
            result=evaluate_mean(policy,raw,names)
            expected=prior(TensorDict({'actor':raw[:1,:63]},batch_size=[1]))[0]
            self.assertTrue(torch.equal(result[0],expected))
            self.assertTrue(torch.equal(result[1],neutral[1]))

    def test_fresh_moving_branch_preserves_standing_after_moving_weights_change(self):
        prior=make_actor(63);prior.eval()
        actor=make_actor(70);critic=make_actor(70)
        scales=initialize_fresh_moving(actor,critic,prior.state_dict());actor.eval()
        raw=torch.randn(8,70);raw[:,63]=torch.tensor([0,.2,0,.2,0,.2,0,.2])
        with torch.no_grad():
            expected=prior(TensorDict({'actor':raw[::2,:63]},batch_size=[4]))
            initial=policy_mean(actor,raw)
            self.assertTrue(torch.equal(initial[::2],expected))
            self.assertTrue(torch.equal(initial[1::2],torch.zeros(4,17)))
            actor.mlp[-1].weight.normal_();actor.mlp[-1].bias.normal_()
            self.assertTrue(torch.equal(policy_mean(actor,raw)[::2],expected))
            self.assertFalse(any(p.requires_grad for p in actor.standing_prior.parameters()))
            self.assertTrue(torch.allclose(actor.obs_normalizer._std+.01,torch.tensor(scales)[None]))
            clone=make_actor(70);initialize_fresh_moving(clone,make_actor(70),prior.state_dict())
            clone.load_state_dict(actor.state_dict(),strict=True);clone.eval()
            self.assertTrue(torch.equal(policy_mean(clone,raw),policy_mean(actor,raw)))

    def test_floor_preserves_mean_and_standing_density_and_kl_after_shuffle(self):
        actor=make_actor(70); actor.eval()
        names=['left_knee_joint','left_hip_pitch_joint','left_ankle_pitch_joint','left_hip_roll_joint']+['waist']*13
        raw=torch.randn(8,70);raw[:,63]=torch.tensor([0,.2,0,.2,0,.2,0,.2])
        obs=TensorDict({'actor':raw},batch_size=[8])
        with torch.no_grad():
            mean=actor(obs).clone();actor(obs,stochastic_output=True);oldstd=actor.output_std.clone()
            install_moving_exploration_floor(actor,names);actor(obs,stochastic_output=True)
            self.assertTrue(torch.equal(actor(obs),mean))
            self.assertTrue(torch.equal(actor.output_std[::2],oldstd[::2]))
            self.assertTrue(torch.allclose(actor.output_std[1,:4],torch.tensor([1.25,.75,.75,.375])))
            saved=tuple(x.clone() for x in actor.output_distribution_params)
            permutation=torch.tensor([7,0,5,2,3,4,1,6])
            actor(TensorDict({'actor':raw[permutation]},batch_size=[8]),stochastic_output=True)
            kl=actor.get_kl_divergence(tuple(x[permutation] for x in saved),actor.output_distribution_params)
            self.assertLess(float(kl.abs().max()),1e-6)


if __name__=='__main__':unittest.main()

"""Backend contracts for pytest and ./geo walk test; never simulation evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

try:
    import mujoco
    import numpy as np
    import scipy
except ImportError as error:
    raise unittest.SkipTest("Optional task-local MuJoCo dependencies are absent") from error
from algorithms.urdf_learn_wasd_walk import model_spec
from algorithms.urdf_learn_wasd_walk.mujoco_backend import Assistance, audit_model, build_model, initialize


class MuJoCoBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled=build_model()

    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path=Path(temporary.name)

    def require(self,name):
        if importlib.util.find_spec(name) is None:
            self.skipTest(f"Optional {name} dependency is absent")

    def test_compiled_asset_mass_frames_and_actions(self):
        compiled = self.compiled

        model, spec, _ = compiled
        result = audit_model(model, spec)
        assert abs(result['total_mass_kg']-1.829753)<1e-12
        assert (model.nq, model.nv, model.nu) == (76, 75, 69)
        assert result['policy_actions'] == 17
        assert result['pd_hold_joints'] == 52
        assert model.neq == 0
        assert model.nmocap == 0
        assert np.array_equal(model.opt.gravity, [0, 0, -9.81])


    def test_nonzero_pose_matches_independent_urdf_fk(self):
        compiled = self.compiled

        model, spec, _ = compiled
        data = initialize(model, spec, pose='geometric')
        pose = {r['name']: float(data.qpos[model.joint(r['name']).qposadr[0]]) for r in spec['joints']}
        transforms, _ = model_spec._joint_world_transforms(ET.parse(model_spec.URDF_PATH).getroot(), pose)
        for name, (rotation, position) in transforms.items():
            b = model.body(name).id
            assert np.allclose(data.xpos[b], np.array(position)+data.qpos[:3], atol=1e-10)
            assert np.allclose(data.xmat[b].reshape(3,3), rotation, atol=1e-10)


    def test_pd_hold_not_weld_and_motor_limits_preserved(self):
        compiled = self.compiled

        model, spec, _ = compiled
        for record in spec['joints']:
            actuator = model.actuator(record['name']).id
            assert np.array_equal(model.actuator_forcerange[actuator], [-record['effort_limit'], record['effort_limit']])
            assert np.array_equal(model.actuator_ctrlrange[actuator], record['limits_rad'])
            assert model.actuator_gainprm[actuator, 0] == record['nominal_stiffness_nm_per_rad']


    def test_audit_rejects_tampered_mass(self):
        compiled = self.compiled

        model, spec, _ = compiled
        b = model.body('root_x').id
        previous = model.body_mass[b]
        try:
            model.body_mass[b] += .1
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                audit_model(model, spec)
        finally:
            model.body_mass[b] = previous


    def test_assistance_has_no_horizontal_force_and_is_bounded(self):

        assist = Assistance(1.)
        wrench = assist.wrench(100, -100, [100, 100, 100], [-100, -100, -100])
        assert np.array_equal(wrench[:2], [0, 0])
        assert np.linalg.norm(wrench[:3]) <= 9.
        assert np.linalg.norm(wrench[3:]) <= 1.+1e-12
        assist.coefficient = 0.
        assert np.array_equal(assist.wrench(100, -100, [100]*3, [-100]*3), np.zeros(6))


    def test_curriculum_requires_competence_reaches_zero_and_rolls_back(self):

        assist = Assistance(1.)
        for _ in range(30):
            assist.update(.95, window_episodes=20)
        assert assist.coefficient == 0.
        assert assist.update(.5, window_episodes=20) == .1
        for _ in range(10):
            assist.update(1., window_episodes=1)
        assert assist.coefficient == .1


    def test_policy_observation_uses_semantic_base_not_rotated_root(self):
        compiled = self.compiled

        self.require('torch')
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import observe
        model, spec, _ = compiled
        data = initialize(model, spec, pose='geometric')
        joints = [model.joint(n).id for n in spec['action_joints']]
        observation = observe(model,data,data.qpos.copy(),joints,np.zeros(17))
        assert observation.shape == (60,)
        assert np.allclose(observation[6:9],[0,0,-1])
        # +Y semantic forward is distinct from the +90-degree mounted pelvis frame.
        data.qvel[:3] = [0,.4,0]
        mujoco.mj_forward(model,data)
        observation = observe(model,data,data.qpos.copy(),joints,np.zeros(17))
        assert np.isclose(observation[1],.4)


    def test_forward_command_observation_and_zero_command_phase(self):

        self.require('torch')
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import StandBatch
        batch=StandBatch(2,42,0.,'forward')
        observation=batch.observations()
        assert observation.shape == (2,65)
        assert np.array_equal(observation[0,60:],np.zeros(5))
        assert np.allclose(observation[1,60:],[.4,0,0,0,1])


    def test_finalizer_rejects_duplicate_runs_before_loading(self):
        tmp_path = self.tmp_path

        self.require('imageio')
        from algorithms.urdf_learn_wasd_walk.mujoco_evidence import finalize
        with self.assertRaisesRegex(ValueError,'distinct'):
            finalize([tmp_path,tmp_path],tmp_path/'review.json',tmp_path/'validation.json')


    def test_finalizer_recomputes_claimed_pass_instead_of_trusting_status(self):
        tmp_path = self.tmp_path

        """Deliberately synthetic failed fixture; never simulation evidence."""
        import json
        self.require('imageio')
        from algorithms.urdf_learn_wasd_walk.mujoco_evidence import finalize
        metrics={'duration_s':30.,'fall_count':1,'reset_count':0,'done_count':0,
                 'max_reference_tilt_rad':0.,'root_height_drop_m':0.,'horizontal_drift_m':0.,
                 'max_abs_action':0.,'max_abs_command':0.,'first_support_exit_time_s':None,
                 'minimum_support_polygon_margin_m':.02,'peak_support_force_body_weight_ratio':1.,
                 'mean_support_force_body_weight_ratio':1.}
        runs=[tmp_path/'a',tmp_path/'b']
        for p in runs:
            p.mkdir(); (p/'dynamics.json').write_text(json.dumps({'milestone':'stand_zero_signal_30s_no_reset',
                        'status':'dynamics_passed_proof_pending','failures':[],'metrics':metrics,'identity':{}}))
        with self.assertRaisesRegex(ValueError,'Recomputed gate failed'):
            finalize(runs,tmp_path/'review.json',tmp_path/'validation.json')
        assert not (tmp_path/'validation.json').exists()


    def test_phase_reward_requires_correct_stance_and_clearance(self):

        self.require('torch')
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import phase_gait_score
        feet={'left_contact':False,'right_contact':True,'left_clearance_m':.012,'right_clearance_m':0.}
        assert np.isclose(phase_gait_score(.25,feet),1.)
        assert phase_gait_score(.75,feet)==0.
        feet['left_contact']=True
        assert phase_gait_score(.25,feet)==0.


    def test_resume_preserves_assistance_progress(self):

        self.require('torch')
        from types import SimpleNamespace
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import restore_curriculum
        batch=SimpleNamespace(assistance=Assistance(1.))
        restore_curriculum(batch,{'curriculum_state':{'coefficient':.2,'successes':2,'completed':[True]*20,'completed_since_update':7}})
        assert batch.assistance.coefficient==.2
        assert batch.assistance.successes==2
        assert len(batch.completed)==20
        assert batch.completed_since_update==7
        assert batch.assistance.update(.95,window_episodes=20)==.1

    def test_forward_exploration_preserves_zero_command_sigma(self):
        self.require('torch')
        import torch
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import ActorCritic
        policy=ActorCritic(65,forward_exploration_multiplier=4.)
        obs=torch.zeros(2,65);obs[1,60]=.4
        distribution=policy.distribution(obs)
        assert torch.allclose(distribution.stddev[0],policy.log_std.exp())
        assert torch.allclose(distribution.stddev[1],4*distribution.stddev[0])
        action=torch.zeros(2,17)
        assert torch.isfinite(distribution.log_prob(action)).all()

    def test_gpu_health_rejects_overflow_nonfinite_and_clock_reset(self):
        from algorithms.urdf_learn_wasd_walk.mujoco_backend import validate_physics_health
        with self.assertRaisesRegex(RuntimeError,'broadphase_pairs capacity'):
            validate_physics_health({'broadphase_pairs':257},{'broadphase_pairs':256},{})
        for name in ('qvel','qacc','actuator_force','constraint_force','xpos','subtree_com'):
            with self.assertRaisesRegex(RuntimeError,'Nonfinite physics array'):
                validate_physics_health({}, {}, {name:np.array([float('nan')])})
        with self.assertRaisesRegex(RuntimeError,'possible reset'):
            validate_physics_health({}, {}, {},clock_delta=-2.)
        validate_physics_health({'contacts':8},{'contacts':256},{'qvel':np.zeros(3)},clock_delta=.002)

    def test_load_reward_supports_weight_transfer_without_rewarding_flight(self):
        self.require('torch')
        import torch
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import phase_load_score
        phase=torch.tensor([.25,.25,.25,.25]);weight=torch.full((4,),18.)
        forces=torch.tensor([[0.,18.],[9.,9.],[3.,15.],[0.,0.]])
        clearance=torch.tensor([[.012,0.],[0.,0.],[.004,0.],[.012,.012]])
        score=phase_load_score(phase,forces,clearance,weight)
        assert torch.isclose(score[0],torch.tensor(1.))
        assert score[2]>score[1]>score[3]
        assert torch.all((score>=0)&(score<=1))

    def test_checkpoint_preserves_bounded_mean_for_evaluation(self):
        self.require('torch')
        import torch
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import ActorCritic, load_actor
        policy=ActorCritic(65,4.,'tanh')
        with torch.no_grad():policy.actor[4].bias.fill_(3.)
        observation=torch.zeros(2,65);observation[:,60]=.2
        expected=policy.actor(observation).detach()
        assert torch.all(expected.abs()<1)
        assert torch.allclose(policy.distribution(observation).mean,expected)
        spec={'source':{'urdf_sha256':'test-urdf','mesh_tree_sha256':'test-mesh'},'action_joints':[]}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'bounded.pt'
            torch.save({'model':policy.state_dict(),'observation_dim':65,'forward_exploration_multiplier':4.,
                        'mean_activation':'tanh','backend':'mujoco_warp_cuda','action_joints':[],
                        'urdf_sha256':'test-urdf','mesh_tree_sha256':'test-mesh'},checkpoint)
            loaded,_=load_actor(checkpoint,None,spec)
            assert loaded.mean_activation=='tanh'
            assert torch.equal(loaded.actor(observation),expected)

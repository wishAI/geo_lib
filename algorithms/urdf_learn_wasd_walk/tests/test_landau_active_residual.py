import copy
import math
import unittest
from types import SimpleNamespace
import torch
from algorithms.urdf_learn_wasd_walk import landau_active_residual as active
from algorithms.urdf_learn_wasd_walk import landau_cruise_residual as cruise
from algorithms.urdf_learn_wasd_walk import landau_active_curriculum as curriculum
from algorithms.urdf_learn_wasd_walk import landau_turn_control as turn
from algorithms.urdf_learn_wasd_walk import landau_teleop_contract as teleop
from algorithms.urdf_learn_wasd_walk.tests.test_landau_cruise_residual import ResidualBindingTests


class ActiveTests(unittest.TestCase):
    def test_early_mixture_preserves_protocol_and_reserved_rows(self):
        ages=torch.arange(90,dtype=torch.float64)%30
        pools=[torch.arange(i*30,(i+1)*30,2) for i in range(3)]
        modes=torch.arange(12000)%3
        torch.manual_seed(17)
        rows=cruise.sample_active_rows(pools,ages,modes,.5)
        self.assertTrue(torch.equal(rows//30,modes))
        self.assertTrue(bool((rows%2==0).all()))
        early=(ages[rows]>=3)&(ages[rows]<=7)
        for mode in range(3):
            fraction=float(early[modes==mode].float().mean())
            self.assertGreater(fraction,.5);self.assertLess(fraction,.65)
            self.assertTrue(bool((ages[rows[modes==mode]]>20).any()))
        with self.assertRaisesRegex(ValueError,'No early'):
            cruise.sample_active_rows([torch.tensor([0])],ages,torch.tensor([0]),.5)

    def test_roll_projection_keeps_control_and_bounds(self):
        from algorithms.urdf_learn_wasd_walk.landau_residual_screen import project_active_delta
        torch.manual_seed(12);baseline=torch.randn(3,3);delta=torch.rand(3,2)*.5-.25
        action=baseline.clone();action[:,[0,2]]+=delta
        names=['left_hip_roll_joint','other','right_hip_roll_joint']
        result=project_active_delta(action,baseline,torch.tensor([1.,0.,.5]),names)
        self.assertTrue(torch.equal(result[0],action[0]));self.assertTrue(torch.equal(result[:,1],baseline[:,1]))
        actual=result-baseline
        self.assertAlmostEqual(float(actual[1,0]),float(actual[1,2]),places=6)
        self.assertTrue(bool(((actual[:,[0,2]]>=delta.min(1).values[:,None]-1e-7)&(actual[:,[0,2]]<=delta.max(1).values[:,None]+1e-7)).all()))

    def test_privileged_critic_keeps_actor_features_and_masks_unavailable_state(self):
        n=3;actor=torch.randn(n,69);original=actor.clone();obs=torch.zeros(n,70)
        mode=torch.tensor([0,1,2]);age=torch.tensor([50.,48.,21.],dtype=torch.float64)
        memory=curriculum.ActiveMemory(n,'cpu');memory.previous_moving[2]=False
        memory.stop_time[2]=20.;memory.stop_xy[2]=torch.tensor([.01,.02]);memory.hold_heading[2]=.1
        teacher=torch.tensor([0.,0.,.1]);positions=torch.zeros(n,3)
        result=cruise.privileged_critic_features(actor,obs,mode,age,positions,torch.eye(3).repeat(n,1,1),memory,teacher,44.)
        self.assertEqual(result.shape,(3,80));self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(torch.equal(actor,original));self.assertTrue(torch.equal(result[:,:69],original))
        # Future M5 absolute target remains visible before stop; missing M6 target is masked.
        self.assertAlmostEqual(float(result[1,73]),-18.,places=5)
        self.assertTrue(torch.equal(result[0,74:78],torch.zeros(4)))
        self.assertAlmostEqual(float(result[2,78]),.5,places=6)
        self.assertEqual(float(result[2,79]),1.)

    def test_critic_expansion_preserves_learned_value_mapping(self):
        def model(width):
            net=torch.nn.Module();net.mlp=torch.nn.Sequential(torch.nn.Linear(width,64),torch.nn.ELU(),
                torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,1));return net
        torch.manual_seed(11);old=model(69);new=model(80)
        new.load_state_dict(cruise.extend_critic_state(old.state_dict(),new.state_dict()))
        value=torch.randn(32,80)
        self.assertTrue(torch.equal(new.mlp[0].weight[:,:69],old.mlp[0].weight))
        self.assertEqual(int(torch.count_nonzero(new.mlp[0].weight[:,69:])),0)
        self.assertTrue(torch.allclose(new.mlp(value),old.mlp(value[:,:69]),atol=1e-6,rtol=0))
        with self.assertRaisesRegex(ValueError,'Unsupported critic'):cruise.extend_critic_state(old.state_dict(),model(81).state_dict())

    def test_long_episodes_complete_stops_without_restart(self):
        modes=torch.tensor([0,0,1,1,2,2]);ages=torch.tensor([3.,84.8,3.,49.8,6.,18.8],dtype=torch.float64)
        ends=cruise.active_episode_end(modes,ages)
        self.assertTrue(torch.allclose(ends,torch.tensor([27.,90.8,55.8,55.8,24.8,24.8],dtype=torch.float64)))
        self.assertTrue(bool(((ends-ages)>=6.-1e-8).all()))
        self.assertTrue(bool((ends[modes==2]<25.).all()))

    def test_commands_match_scalar_boundaries(self):
        turn.configure_profile(44.)
        ages=torch.tensor([2.98,3.,3.02,5.98,6.,6.02,17.98,18.,18.02,19.98,20.,20.02,46.98,47.,47.02,48.98,49.,49.02],dtype=torch.float64)
        for mode in (1,2):
            actual=curriculum.commands(torch.full((len(ages),),mode),ages,torch.zeros(len(ages),2),torch.zeros(len(ages)),44.)
            expected=[]
            for age in ages.tolist():
                if mode==1:
                    f,y,_=turn.command_profile(age);expected.append((f,0.,y))
                else:expected.append(teleop.command_profile(age))
            self.assertTrue(torch.allclose(actual,torch.tensor(expected,dtype=torch.float64),atol=1e-12,rtol=0))

    def test_async_memory_and_holding_prior_match_scalar(self):
        turn.configure_profile(44.)
        # Both fade edges, both stop edges and already-held states.
        starts=[46.8,48.8,49.2,17.8,19.8,20.2];modes=torch.tensor([1,1,1,2,2,2])
        memory=curriculum.ActiveMemory(6,'cpu');scalars=[turn.CommandMemory() for _ in starts]
        obs=torch.zeros(6,70);xyz=torch.zeros(6,3);rotation=torch.eye(3).repeat(6,1,1)
        def data(age):
            result=torch.zeros(1,70);result[:,62]=1.-age/30.;result[:,68]=math.sin(.03*age);result[:,69]=math.cos(.03*age)-1.
            pos=torch.tensor([[.001*age,.002*age,.3]])
            return result,pos
        for i,(start,mode) in enumerate(zip(starts,modes)):
            for step in range(round(start/.02)+1):
                age=step*.02;single,pos=data(age)
                if mode==1:f,y,_=turn.command_profile(age);single[:,63:66]=torch.tensor([f,0.,y])
                else:single[:,63:66]=torch.tensor(teleop.command_profile(age))
                scalars[i].observe(single,pos,rotation[i:i+1],age)
            scalar=scalars[i]
            memory._reference[i]=scalar.reference;memory._blend[i]=scalar.left_cruise_blend;memory._feedback[i]=scalar.left_feedback_blend
            memory.previous_yaw[i]=scalar.previous_yaw;memory.previous_moving[i]=scalar.previous_moving
            memory.left_cruise_start[i]=scalar.left_cruise_start if scalar.left_cruise_start is not None else float('inf')
            if scalar.anchor_xy is not None:memory.stop_xy[i]=scalar.anchor_xy[0];memory.stop_time[i]=scalar.anchor_time
        for step in range(1,70):
            ages=torch.tensor(starts,dtype=torch.float64)+step*.02
            for i,age in enumerate(ages.tolist()):obs[i],xyz[i]=[v[0] for v in data(age)]
            obs[:,63:66]=curriculum.commands(modes,ages,xyz[:,:2].double(),torch.zeros(6),44.).float()
            memory.advance_command(obs,xyz,ages,.02)
            prior=memory.prior(obs,xyz,rotation,ages)
            for i,scalar in enumerate(scalars):
                _,expected=scalar.observe(obs[i:i+1],xyz[i:i+1],rotation[i:i+1],float(ages[i]))
                self.assertAlmostEqual(float(memory._reference[i]),scalar.reference,places=10)
                self.assertAlmostEqual(float(memory._blend[i]),scalar.left_cruise_blend,places=10)
                self.assertAlmostEqual(float(memory._feedback[i]),scalar.left_feedback_blend,places=10)
                self.assertTrue(torch.allclose(prior[i],expected[0],atol=2e-7,rtol=0))
        self.assertTrue(torch.equal(memory.left_cruise_blend,torch.zeros(6)))

    def test_zero_active_exact_and_combined_bound(self):
        torch.manual_seed(4);n=100
        memory=SimpleNamespace(left_cruise_blend=torch.linspace(0,1,n),left_feedback_blend=torch.rand(n),reference=0.)
        obs=torch.zeros(n,70);obs[:,63]=.2;names=['left_hip_roll_joint','other','right_hip_roll_joint']
        base=torch.randn(n,3);latent=torch.randn(n,2)*10
        frozen=cruise.compose(base,latent,memory,names,obs)
        self.assertTrue(torch.equal(active.compose(frozen,torch.zeros_like(latent),memory,names,obs),frozen))
        result=active.compose(frozen,latent,memory,names,obs)
        self.assertLessEqual(float(((result-base)*.08).abs().max()),.020001)
        obs[:,63]=0.
        self.assertTrue(torch.equal(active.compose(frozen,torch.full_like(latent,float('nan')),memory,names,obs),frozen))


class ActiveBindingTests(ResidualBindingTests):
    # Inherit the cruise-only rejection cases, then extend their fixture explicitly.
    def enable(self):
        from pathlib import Path
        g=self.gates;directory=self.root/'cruise';directory.mkdir();cp=directory/'model.pt'
        torch.save(self.blob,cp);(directory/'residual_source.py').write_text('residual')
        meta={**self.meta,'status':'complete','checkpoints':{cp.name:g.digest(cp)}}
        g.write(directory/'training.json',meta)
        self.meta.update(residual_variant=active.VARIANT,active_residual_bound_rad=.02,active_residual_feature_dim=69,
            active_residual_gate=active.GATE,active_residual_action_joints=['left_hip_roll_joint','right_hip_roll_joint'],
            frozen_cruise_checkpoint=str(cp),frozen_cruise_checkpoint_sha256=g.digest(cp),frozen_cruise_training_sha256=g.digest(directory/'training.json'))
        for folder in (self.folder,self.candidate_dir):(folder/'active_residual_source.py').write_text('active')
        self.meta['active_residual_source_sha256']=g.digest(self.folder/'active_residual_source.py')
        self.result.update(residual_variant=active.VARIANT,residual_dispatch=active.DISPATCH,active_residual_source_sha256=self.meta['active_residual_source_sha256'])
        self.blob.update(residual_variant=active.VARIANT,active_residual_actor_state_dict={'weight':torch.zeros(2,2)})
        torch.save(self.blob,self.checkpoint)
    def test_valid_active(self):self.enable();self.check();self.assertTrue(active.check_contract(self.meta,self.blob,self.gates.digest))
    def test_unexpected_head_rejected(self):
        self.blob['active_residual_actor_state_dict']={};torch.save(self.blob,self.checkpoint)
        with self.assertRaisesRegex(ValueError,'Unexpected active'):self.check()
        with self.assertRaisesRegex(ValueError,'Unexpected active'):active.check_contract(self.meta,self.blob,self.gates.digest)
    def test_missing_head_rejected(self):
        self.enable();del self.blob['active_residual_actor_state_dict'];torch.save(self.blob,self.checkpoint)
        with self.assertRaisesRegex(ValueError,'Missing active'):self.check()
    def test_changed_cruise_rejected(self):
        self.enable();self.blob['residual_actor_state_dict']['weight']+=.1;torch.save(self.blob,self.checkpoint)
        with self.assertRaisesRegex(ValueError,'Frozen cruise weights'):self.check()
        with self.assertRaisesRegex(ValueError,'Frozen cruise weights'):active.check_contract(self.meta,self.blob,self.gates.digest)
    def test_missing_active_evidence_rejected(self):
        self.enable();del self.result['active_residual_source_sha256']
        with self.assertRaisesRegex(ValueError,'Active residual source mismatch'):self.check()
    def test_variant_mismatch_rejected(self):
        self.enable();del self.blob['residual_variant'];torch.save(self.blob,self.checkpoint)
        with self.assertRaisesRegex(ValueError,'variant mismatch'):self.check()


class AdaptationBindingTests(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        from algorithms.urdf_learn_wasd_walk import landau_cruise_adaptation as adaptation
        self.adaptation=adaptation
        ResidualBindingTests.setUp(self);ActiveBindingTests.enable(self)
        g=self.gates;cp=Path(self.meta['frozen_cruise_checkpoint'])
        for folder in (self.parent_dir,self.candidate_dir,cp.parent):(folder/'turn_source.py').write_text('turn')
        turn_hash=g.digest(self.parent_dir/'turn_source.py');self.parent_meta['turn_source_sha256']=turn_hash
        g.write(self.parent_dir/'training.json',self.parent_meta)
        self.meta.update(turn_source_sha256=turn_hash,parent_training_sha256=g.digest(self.parent_dir/'training.json'))
        prior=g.read(cp.parent/'training.json');prior.update(turn_source_sha256=turn_hash,parent_training_sha256=self.meta['parent_training_sha256'])
        g.write(cp.parent/'training.json',prior);self.meta['frozen_cruise_training_sha256']=g.digest(cp.parent/'training.json')
        directory=self.root/'donor';directory.mkdir();self.donor=directory/'model.pt'
        torch.save(self.blob,self.donor)
        for name in ('control_source.py','turn_source.py','residual_source.py','active_residual_source.py'):
            (directory/name).write_bytes((self.candidate_dir/name).read_bytes())
        old=copy.deepcopy(self.meta);old.update(status='complete',checkpoints={self.donor.name:g.digest(self.donor)})
        g.write(directory/'training.json',old)
        self.meta={k:v for k,v in self.meta.items() if not k.startswith('frozen_cruise_')}
        adaptation.metadata(self.meta,self.donor,old,cp,g.digest)
        self.blob['residual_variant']=adaptation.VARIANT;self.result['residual_variant']=adaptation.VARIANT
        torch.save(self.blob,self.checkpoint)

    def check_both(self):
        torch.save(self.blob,self.checkpoint)
        self.gates.check_residual_source(self.folder,self.result,self.meta,self.checkpoint)
        self.assertTrue(self.adaptation.check_contract(self.meta,self.blob,self.gates.digest))

    def test_changed_cruise_allowed_with_active_frozen(self):
        self.blob['residual_actor_state_dict']['weight']+=.2;self.check_both()

    def test_changed_active_rejected(self):
        self.blob['active_residual_actor_state_dict']['weight']+=.1
        torch.save(self.blob,self.checkpoint)
        with self.assertRaisesRegex(ValueError,'Frozen active weights'):self.gates.check_residual_source(self.folder,self.result,self.meta,self.checkpoint)
        with self.assertRaisesRegex(ValueError,'Frozen active weights'):self.adaptation.check_contract(self.meta,self.blob,self.gates.digest)

    def test_changed_donor_rejected(self):
        with self.donor.open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen active input'):self.check_both()
        with self.assertRaisesRegex(ValueError,'Frozen active input'):self.adaptation.check_contract(self.meta,self.blob,self.gates.digest)

    def test_cannot_claim_frozen_cruise(self):
        self.meta['frozen_cruise_checkpoint']='misleading'
        with self.assertRaisesRegex(ValueError,'frozen'):self.check_both()
        with self.assertRaisesRegex(ValueError,'frozen'):self.adaptation.check_contract(self.meta,self.blob,self.gates.digest)

    def test_cruise_change_inactive_is_exact(self):
        n=20;memory=SimpleNamespace(left_cruise_blend=torch.zeros(n),left_feedback_blend=torch.ones(n),reference=0.)
        obs=torch.zeros(n,70);obs[:,63]=.2;names=['left_hip_roll_joint','right_hip_roll_joint']
        base=torch.randn(n,2);fixed=torch.randn(n,2)
        a=active.compose(cruise.compose(base,torch.randn(n,2),memory,names,obs),fixed,memory,names,obs)
        b=active.compose(cruise.compose(base,torch.randn(n,2)*10,memory,names,obs),fixed,memory,names,obs)
        self.assertTrue(torch.equal(a,b))


if __name__=='__main__':unittest.main()

"""Check model capability and bind compatibility independently of training."""
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from algorithms.urdf_learn_wasd_walk import model_spec as source, leg_model_variant as variant


class AnatomicalLegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=ET.parse(source.URDF_PATH).getroot();cls.new=ET.parse(variant.URDF).getroot()
        cls.old_world,_=source._joint_world_transforms(cls.old)
        cls.world,_=source._joint_world_transforms(cls.new)
        cls.spec=variant.specification()

    def test_preserves_every_existing_bind_and_total_mass(self):
        self.assertEqual(source._sha256(source.URDF_PATH),source.EXPECTED_URDF_SHA256)
        for name,old in self.old_world.items():
            np.testing.assert_allclose(self.world[name][0],old[0],atol=1e-12)
            np.testing.assert_allclose(self.world[name][1],old[1],atol=1e-12)
        self.assertAlmostEqual(sum(float(i.get('value')) for i in self.new.findall('./link/inertial/mass')),18.,places=10)
        self.assertEqual(len(self.spec['action_joints']),19)
        self.assertFalse(set(source.FINGER_JOINTS)&set(self.spec['action_joints']))

    def test_ankle_roll_adds_missing_foot_twist_direction(self):
        records={r['name']:r for r in self.spec['joints']}
        for side,suffix in [('left','l'),('right','r')]:
            columns=[];pivot=np.array(self.world['foot_'+suffix][1])
            for role in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']:
                record=records[f'{side}_{role}_joint'];j=self.new.find(f"joint[@name='{record['name']}']")
                frame=source._compose(self.world[j.find('parent').get('link')],source._origin_transform(j.find('origin')))
                axis=np.array(record['axis_world_zero_pose']);columns.append(np.r_[np.cross(axis,pivot-frame[1]),axis])
            self.assertEqual(np.linalg.matrix_rank(np.array(columns[:-1]).T,tol=1e-6),5)
            self.assertEqual(np.linalg.matrix_rank(np.array(columns).T,tol=1e-6),6)
            for role,wanted in [('hip_roll',[0,1,0]),('hip_yaw',[0,0,1]),('ankle_roll',[0,1,0])]:
                np.testing.assert_allclose(records[f'{side}_{role}_joint']['axis_world_zero_pose'],wanted,atol=1e-8)
            # The bind pose already bends the knee by ~14.5 deg: extension is negative.
            knee=records[f'{side}_knee_joint']['limits_rad']
            self.assertGreaterEqual(knee[0]+.25353,0)
            self.assertLess(knee[0]+.25353,.01)
            self.assertLess(knee[1]+.25353,2.4)

    def test_virtual_link_partitions_original_foot_inertia(self):
        scale=self.spec['variant']['density_scale']
        for side,suffix in [('left','l'),('right','r')]:
            old=self.old.find(f"link[@name='foot_{suffix}']/inertial")
            pieces=[self.new.find(f"link[@name='{name}']/inertial") for name in [f'foot_{suffix}',f'{side}_ankle_pitch_link']]
            self.assertAlmostEqual(sum(float(p.find('mass').get('value')) for p in pieces),float(old.find('mass').get('value'))*scale,places=12)
            for k,v in old.find('inertia').attrib.items():
                self.assertAlmostEqual(sum(float(p.find('inertia').get(k)) for p in pieces),float(v)*scale,places=12)
            self.assertEqual(pieces[0].find('origin').attrib,pieces[1].find('origin').attrib)

if __name__=='__main__':unittest.main()

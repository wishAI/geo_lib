import unittest
import xml.etree.ElementTree as ET
from algorithms.urdf_learn_wasd_walk.continuation import redistribute_mass


class MassVariantTests(unittest.TestCase):
    def test_mass_is_conserved_and_inertia_scales_with_density(self):
        root = ET.Element('mujoco')
        names = [f'thumb_{i}' for i in range(38)] + ['root_x', 'spine_01_x', 'spine_02_x', 'spine_03_x', 'head_x']
        for name in names:
            b = ET.SubElement(root, 'body', name=name)
            ET.SubElement(b, 'inertial', mass='.02', pos='0 0 0', fullinertia='.003 .004 .005 0 0 0')
        xml = ET.tostring(root, encoding='unicode')
        changed, audit = redistribute_mass(xml, 'balanced_hands_v1')
        self.assertAlmostEqual(audit['total_mass_before_kg'], audit['total_mass_after_kg'])
        for body in ET.fromstring(changed).iter('body'):
            i = body.find('inertial')
            factor = float(i.get('mass')) / .02
            self.assertAlmostEqual(float(i.get('fullinertia').split()[0]), .003 * factor)
            self.assertEqual(i.get('pos'), '0 0 0')
        self.assertNotIn('head_x', audit['changes'])
        self.assertEqual(redistribute_mass(xml, 'original')[0], xml)

    def test_unknown_model_or_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            redistribute_mass('<mujoco/>', 'balanced_hands_v1')
        with self.assertRaises(ValueError):
            redistribute_mass('<mujoco/>', 'unbounded')

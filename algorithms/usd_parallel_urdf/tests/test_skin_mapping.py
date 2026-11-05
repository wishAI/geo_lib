"""Skin regression with deliberately different USD and serial URDF hierarchies."""
import sys
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skeleton_common import UrdfSkinMapper, world_matrices_from_local
from g1_pose import world_map_from_joint_specs


class SkinMappingTests(unittest.TestCase):
    def exercise_chain(self, part):
        for serial in (False, True):
            for side in ('l', 'r'):
                names = ['root_x', f'arm_stretch_{side}', f'arm_twist_{side}',
                         f'forearm_stretch_{side}', f'forearm_twist_{side}', f'hand_{side}']
                if part=='thigh':
                    names=[n.replace('forearm','leg').replace('arm','thigh').replace('hand','foot') for n in names]
                elif part=='leg':
                    names=[n.replace('forearm_stretch','foot').replace('forearm_twist','toes_01').replace('arm','leg').replace('hand','toe_tip') for n in names]
                parents = [-1, 0, 1, 1, 3, 3]
                positions = [(0, 0, .4), (.1, 0, 0), (0, 0, 0), (0, .1, 0), (0, .05, 0), (0, .1, 0)]
                records = []
                for i, (name, parent, xyz) in enumerate(zip(names, parents, positions)):
                    local = np.eye(4)
                    local[:3, 3] = xyz
                    # A nontrivial source bind basis must survive zero-pose mapping.
                    if i == 0:
                        local[:3, :3] = [[0, -1.001, 0], [1.001, 0, 0], [0, 0, 1.001]]
                    records.append(dict(index=i, name=name, parent_index=parent, local_matrix=local))
                skin_rest = world_matrices_from_local(records, [r['local_matrix'] for r in records])
                xml = ET.Element('robot', name='skin_fixture')
                for i, name in enumerate(names):
                    parent = parents[i]
                    xyz = positions[i]
                    if serial and i == 3:
                        parent = 2
                    if serial and i == 5:
                        parent = 4
                        xyz = (0, .05, 0)
                    joint = ET.SubElement(xml, 'joint', name=f'joint_{i}', type='fixed' if i == 0 else 'revolute')
                    ET.SubElement(joint, 'parent', link='base_link' if parent < 0 else names[parent])
                    ET.SubElement(joint, 'child', link=name)
                    ET.SubElement(joint, 'origin', xyz=' '.join(map(str, xyz)), rpy='0 0 0')
                    ET.SubElement(joint, 'axis', xyz='0 0 1' if i == 2 else '1 0 0')
                if serial and part=='thigh':
                    # Virtual hip/ankle nodes are absent from the USD skin.
                    for index,label in [(2,'hip_roll'),(5,'ankle_pitch')]:
                        downstream=xml.find(f"joint[@name='joint_{index}']")
                        parent=downstream.find('parent').get('link')
                        extra=ET.SubElement(xml,'joint',name=label,type='revolute')
                        ET.SubElement(extra,'parent',link=parent)
                        ET.SubElement(extra,'child',link=label+'_virtual')
                        ET.SubElement(extra,'origin',xyz='0 0 0',rpy='0 0 0')
                        ET.SubElement(extra,'axis',xyz='0 1 0')
                        downstream.find('parent').set('link',label+'_virtual')
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / 'robot.urdf'
                    ET.ElementTree(xml).write(path)
                    mapper = UrdfSkinMapper(records, path)
                rest = world_map_from_joint_specs(mapper.specs)
                for actual, record in zip(mapper.apply({}), records):
                    np.testing.assert_allclose(actual, record['local_matrix'], atol=1e-12)
                for angle in (-1.2, -.5, .5, 1.2):
                    pose = {names[1]: .4, names[2]: angle, names[3]: .8, names[4]: -angle, names[5]: .2}
                    worlds = world_map_from_joint_specs(mapper.specs, {f'joint_{i}': pose.get(n, 0) for i, n in enumerate(names)})
                    actual = world_matrices_from_local(records, mapper.apply(pose))
                    for i, name in enumerate(names):
                        owner = names[2] if serial and i == 1 else (names[4] if serial and i==3 and part=='thigh' else name)
                        expected = worlds[owner] @ np.linalg.inv(rest[owner]) @ skin_rest[i]
                        np.testing.assert_allclose(actual[i], expected, atol=1e-12)
                    if serial:
                        # A point on the upper sleeve at the rest elbow must reach
                        # the articulated elbow, even for a transverse twist axis.
                        delta = actual[1] @ np.linalg.inv(skin_rest[1])
                        np.testing.assert_allclose(delta @ rest[names[3]][:, 3], worlds[names[3]][:, 3], atol=1e-12)

    def test_serial_motion_and_geometry_ownership(self):
        for part in ('arm','thigh','leg'):
            with self.subTest(part=part):self.exercise_chain(part)


if __name__ == '__main__':
    unittest.main()

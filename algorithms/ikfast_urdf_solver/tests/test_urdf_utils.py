from __future__ import annotations

from pathlib import Path

import mujoco

from algorithms.ikfast_urdf_solver.urdf_utils import resolve_chain, sanitize_urdf_for_mujoco


def test_resolve_chain_handles_active_and_fixed_joints(tmp_path: Path) -> None:
    urdf_path = tmp_path / "chain.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="chain">
  <link name="base"/>
  <link name="mid"/>
  <link name="tool_mount"/>
  <link name="tool"/>
  <joint name="joint_a" type="revolute">
    <parent link="base"/>
    <child link="mid"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>
  <joint name="mount_joint" type="fixed">
    <parent link="mid"/>
    <child link="tool_mount"/>
  </joint>
  <joint name="joint_b" type="continuous">
    <parent link="tool_mount"/>
    <child link="tool"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    chain = resolve_chain(urdf_path, "base", "tool")
    assert chain.active_joint_names == ("joint_a", "joint_b")
    assert chain.active_joint_limits[0].lower == -1.0
    assert chain.active_joint_limits[1].continuous is True


def test_sanitize_urdf_for_mujoco_drops_mesh_dependence(tmp_path: Path) -> None:
    src = tmp_path / "mesh_robot.urdf"
    dst = tmp_path / "kinematics_only.urdf"
    src.write_text(
        """<?xml version="1.0"?>
<robot name="mesh_robot">
  <link name="base">
    <visual>
      <geometry><mesh filename="package://missing/base.stl"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="package://missing/base.stl"/></geometry>
    </collision>
  </link>
  <link name="tip"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="tip"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>
  <ros2_control name="ignored" type="system"/>
</robot>
""",
        encoding="utf-8",
    )
    sanitize_urdf_for_mujoco(src, dst)
    model = mujoco.MjModel.from_xml_path(str(dst))
    assert model.njnt == 1

from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from algorithms.urdf_learn_wasd_walk import model_spec
from algorithms.urdf_learn_wasd_walk import policy_stand_initialization as init


class CloneArray(np.ndarray):
    def clone(self):
        return self.copy()

    def detach(self):
        return self

    def cpu(self):
        return self

    def zero_(self):
        self[:] = 0

    def repeat(self, n, dim):
        return np.tile(np.asarray(self), (n, dim)).view(CloneArray)

    def abs(self):
        return np.abs(self)


def tensor(values):
    return np.asarray(values, dtype=float).view(CloneArray)


class PolicyInitializationTests(unittest.TestCase):
    def test_early_failure_preserves_identity_without_claiming_runtime_validation(self):
        from algorithms.urdf_learn_wasd_walk import policy_stand

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output_dir=root, mode="train", smoke=False,
                                   runtime_stage="fixed_root_gravity_settling")
            with patch.object(policy_stand.contract, "safe_output_dir", return_value=root), \
                 patch.object(policy_stand, "REPO_BOOTSTRAP_ROOT", root), \
                 patch.object(policy_stand, "_source_commit", return_value="test-commit"):
                policy_stand._write_failure(args, ValueError("unsettled joint"), "test traceback")
            failure = json.loads((root / "train_failure.json").read_text())
            self.assertEqual(failure["lineage"], policy_stand.contract.LINEAGE)
            self.assertFalse(failure["gate_eligible"])
            self.assertFalse(failure["input"]["identity_is_runtime_verified"])
            self.assertEqual(failure["input"]["expected_mesh_tree_sha256"], model_spec.EXPECTED_MESH_TREE_SHA256)
            self.assertEqual(failure["initialization_protocol"], init.protocol())
            self.assertEqual(failure["runtime_stage"], args.runtime_stage)
            self.assertNotIn("checkpoint", failure)
            self.assertEqual((root / "train_traceback.log").read_text(), "test traceback")

    def test_missing_old_or_partial_initialization_cannot_authorize_checkpoint(self):
        prior, source = {"sha256": "parent"}, {"mesh_tree_sha256": "current"}
        report = {**init.protocol(), "status": "initialized_not_validated",
                  "policy_steps_during_settling": 0, "prior_gate": prior, "input": source,
                  "environments": [{"env_id": 0, "support_margin_m": .01}]}
        init.validate_report(report, num_envs=1, prior=prior, source=source)
        for changed in ({}, {**report, "fixed_root_steps": 1000},
                        {**report, "policy_steps_during_settling": 200},
                        {**report, "prior_gate": {}}, {**report, "environments": []},
                        {**report, "environments": [{"env_id": 0, "support_margin_m": math.nan}]}):
            with self.assertRaises(ValueError):
                init.validate_report(changed, num_envs=1, prior=prior, source=source)

    def test_batched_settling_preserves_episode_clock_and_updates_action_offset(self):
        names = ["left_hip_pitch_joint", "right_hip_pitch_joint"]
        data = SimpleNamespace(
            default_root_state=tensor([[0, 0, .01, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 2),
            default_joint_pos=tensor([[0, 0]] * 2), default_joint_vel=tensor([[0, 0]] * 2),
            joint_stiffness=tensor([[10, 10]] * 2), joint_pos=tensor([[0, 0]] * 2),
            applied_torque=tensor([[.2, -.2]] * 2), joint_vel=tensor([[.01, .01]] * 2),
            joint_pos_limits=tensor([[[-1, 1], [-1, 1]]] * 2),
            joint_effort_limits=tensor([[50, 50]] * 2),
        )
        writes = []
        def write_joint(q, v, **kwargs):
            data.joint_pos[:] = q
        robot = SimpleNamespace(data=data, joint_names=names,
                                write_joint_state_to_sim=write_joint,
                                write_root_state_to_sim=lambda *a, **k: None,
                                set_joint_position_target=lambda v, **k: writes.append(v.copy()))
        class Scene(dict):
            env_origins = tensor([[0, 0, 0], [2, 3, 0]])
            updates = 0
            def write_data_to_sim(self):
                pass
            def update(self, dt):
                self.updates += 1
                # A deterministic fake equilibrium, different for each clone.
                data.joint_pos[:] = [[.1, -.1], [.11, -.09]]
        class Sim:
            steps = 0
            def step(self, *, render):
                assert render is False
                self.steps += 1
        action = SimpleNamespace(_joint_names=names, _joint_ids=[0, 1], _offset=tensor([[0, 0]] * 2))
        env = SimpleNamespace(scene=Scene(robot=robot), sim=Sim(), device="cpu", num_envs=2,
                              physics_dt=.002, episode_length_buf=[0, 0], common_step_counter=0,
                              action_manager=SimpleNamespace(get_term=lambda n: action))
        torch = SimpleNamespace(tensor=lambda v, **k: tensor(v), zeros_like=lambda v: tensor(np.zeros_like(v)),
                                abs=np.abs, all=np.all, isfinite=np.isfinite)
        seed = {"joint_positions_rad": dict(zip(names, [.1, -.1])),
                "fixed_root_gravity_torque_nm": dict(zip(names, [-.2, .2])),
                "ground_aligned_base_z_m": .02}
        spec = {"nominal_pose": {"joint_positions_rad": seed["joint_positions_rad"],
                                 "base_position_m": [0, 0, .01]}, "source": {"mesh_tree_sha256": "test"}}
        geometry = {"ground_aligned_base_z_m": .03, "support_margin_m": .04,
                    "support_hull_xy_m": [[-1, -1], [1, -1], [0, 1]]}
        with patch.dict("sys.modules", {"torch": torch}), \
             patch.object(model_spec, "ACTION_JOINTS", names), \
             patch.object(model_spec, "build_robot_spec", return_value=spec), \
             patch.object(model_spec, "derive_static_pose", return_value=seed), \
             patch.object(model_spec, "analyze_pose_geometry", return_value=geometry):
            report = init.initialize(env, {"sha256": "parent"})
            with self.assertRaisesRegex(RuntimeError, "only once"):
                init.initialize(env, {})
        self.assertEqual(env.sim.steps, 2000)
        self.assertEqual(env.scene.updates, 2000)
        self.assertEqual(env.episode_length_buf, [0, 0])
        self.assertEqual(env.common_step_counter, 0)
        np.testing.assert_allclose(writes[0], [[.12, -.12]] * 2)
        np.testing.assert_allclose(data.default_joint_pos, [[.1, -.1], [.11, -.09]])
        np.testing.assert_allclose(action._offset, [[.12, -.12], [.13, -.11]])
        np.testing.assert_allclose(data.default_root_state[:, 2], [.03, .03])
        self.assertEqual(report["prior_gate"]["sha256"], "parent")
        self.assertEqual(len(report["environments"]), 2)

    def audit(self, name="left_hip_pitch_joint", q=0.1, torque=0.2, speed=0.01,
              stiffness=10.0, bounds=(-0.7, 0.7), effort=50.0):
        return init.audit_release([name], [q], [torque], [speed], [stiffness],
                                  [bounds], [effort])[0]

    def test_preload_opposes_measured_gravity_at_release(self):
        record = self.audit()
        self.assertAlmostEqual(record["released_target_rad"], 0.12)
        self.assertAlmostEqual(10 * (record["released_target_rad"] - 0.1), 0.2)
        self.assertAlmostEqual(self.audit(torque=-0.2)["released_target_rad"], 0.08)

    def test_exact_finger_tolerance_preserves_raw_and_clamped_values(self):
        finger = model_spec.FINGER_JOINTS[0]
        tol = model_spec.DERIVED_POSE_FINGER_LIMIT_TOLERANCE_RAD
        for value in (0.0, tol - 1e-10, tol):
            r = self.audit(name=finger, q=value, torque=0, bounds=(-1, 0))
            self.assertEqual(r["raw_settled_position_rad"], value)
            self.assertEqual(r["settled_position_rad"], 0)
            self.assertEqual(r["released_target_rad"], 0)
        with self.assertRaisesRegex(ValueError, "limits"):
            self.audit(name=finger, q=tol + 1e-10, torque=0, bounds=(-1, 0))
        with self.assertRaisesRegex(ValueError, "limits"):
            self.audit(q=1e-12, torque=0, bounds=(-1, 0))

    def test_bad_mechanics_fail_before_free_root_release(self):
        for kwargs in ({"q": math.nan}, {"torque": math.inf}, {"stiffness": 0},
                       {"speed": 0.5001}, {"effort": 0}, {"torque": 50.01},
                       {"bounds": (1, -1)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.audit(**kwargs)
        with self.assertRaisesRegex(ValueError, "aligned"):
            init.audit_release(["a", "a"], [0], [0], [0], [1], [[-1, 1]], [1])

    def test_four_second_settling_is_outside_policy_episode(self):
        p = init.protocol()
        self.assertEqual(p["fixed_root_steps"], 2000)
        self.assertEqual(p["fixed_root_duration_s"], 4.0)
        self.assertEqual(p["averaging_duration_s"], 0.5)
        self.assertFalse(p["settling_counts_toward_policy_duration"])
        self.assertFalse(p["fixed_root_contact_load_is_gating"])

    def test_partial_reset_restores_locked_joints_and_preload_without_stepping(self):
        calls = {}
        data = SimpleNamespace(
            default_root_state=tensor([[0, 0, 0.01, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 2),
            default_joint_pos=tensor([[0.1, -0.2], [0.3, -0.4]]),
            default_joint_vel=tensor([[0, 0], [0, 0]]),
        )
        def root_write(value, env_ids):
            calls["root"] = value.copy()
            calls["ids"] = env_ids
        def joint_write(q, v, env_ids):
            calls["q"], calls["v"] = q.copy(), v.copy()
        def target_write(value, env_ids):
            calls["targets"] = value.copy()
        robot = SimpleNamespace(data=data, write_root_state_to_sim=root_write,
                                write_joint_state_to_sim=joint_write,
                                set_joint_position_target=target_write)
        class Scene(dict):
            env_origins = tensor([[0, 0, 0], [2, 3, 0]])
        env = SimpleNamespace(scene=Scene(robot=robot),
                              _landau_settled_targets=tensor([[0.11, -0.22], [0.33, -0.44]]))
        init.restore_settled_baseline(env, [1])
        self.assertEqual(calls["ids"], [1])
        np.testing.assert_allclose(calls["root"][0, :3], [2, 3, 0.01])
        np.testing.assert_allclose(calls["q"], [[0.3, -0.4]])
        np.testing.assert_allclose(calls["targets"], [[0.33, -0.44]])
        np.testing.assert_allclose(data.default_root_state[:, :3], [[0, 0, 0.01]] * 2)
        # No simulator exists in this fake: asynchronous reset cannot advance physics.
        del env._landau_settled_targets
        with self.assertRaisesRegex(RuntimeError, "before gravity"):
            init.restore_settled_baseline(env, [1])


if __name__ == "__main__":
    unittest.main()

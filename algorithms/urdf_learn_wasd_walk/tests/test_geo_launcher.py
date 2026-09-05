from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_geo_module():
    name = "geo_launcher_contract_test"
    loader = importlib.machinery.SourceFileLoader(name, str(REPO_ROOT / "geo"))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


class GeoLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geo = _load_geo_module()

    def test_logged_child_failure_preserves_stderr_and_machine_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failure = root / "failure.json"
            console = root / "console.log"
            success = root / "success.json"
            launch = self.geo.LaunchSpec(
                "direct",
                [
                    sys.executable,
                    "-c",
                    "import sys; print('preserved-stderr-marker', file=sys.stderr); raise SystemExit(7)",
                ],
                success_artifact=success,
                failure_artifact=failure,
                console_log=console,
            )
            returncode = self.geo._run_with_runner(launch, dry_run=False, verbose=False)
            self.assertEqual(returncode, 7)
            self.assertIn("preserved-stderr-marker", console.read_text(encoding="utf-8"))
            evidence = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "launcher_child_failed")
            self.assertEqual(evidence["returncode"], 7)
            self.assertIsNotNone(evidence["console_log_sha256"])

    def test_launcher_augments_child_failure_without_erasing_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failure = root / "failure.json"
            console = root / "console.log"
            script = (
                "import json, pathlib, sys; "
                "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
                "'status':'failed_to_execute','exception':{'traceback':'kept'}})); "
                "raise SystemExit(3)"
            )
            launch = self.geo.LaunchSpec(
                "direct",
                [sys.executable, "-c", script, str(failure)],
                success_artifact=root / "success.json",
                failure_artifact=failure,
                console_log=console,
            )
            self.assertEqual(
                self.geo._run_with_runner(launch, dry_run=False, verbose=False), 3
            )
            evidence = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed_to_execute")
            self.assertEqual(evidence["exception"]["traceback"], "kept")
            self.assertEqual(evidence["launcher"]["returncode"], 3)
            self.assertIsNotNone(evidence["launcher"]["console_log_sha256"])

    def test_failed_gate_artifact_is_not_mislabeled_as_launcher_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            success = root / "validation.json"
            failure = root / "failure.json"
            script = (
                "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('failed gate'); "
                "raise SystemExit(1)"
            )
            launch = self.geo.LaunchSpec(
                "direct",
                [sys.executable, "-c", script, str(success)],
                success_artifact=success,
                failure_artifact=failure,
                console_log=root / "console.log",
            )
            self.assertEqual(
                self.geo._run_with_runner(launch, dry_run=False, verbose=False), 1
            )
            self.assertTrue(success.is_file())
            self.assertFalse(failure.exists())

    def test_required_artifact_status_fails_closed_when_shell_masks_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "probe.json"
            script = (
                "import json, pathlib, sys; "
                "pathlib.Path(sys.argv[1]).write_text(json.dumps({'status':'failed'}))"
            )
            launch = self.geo.LaunchSpec(
                "direct",
                [sys.executable, "-c", script, str(artifact)],
                success_artifact=artifact,
                failure_artifact=root / "failure.json",
                required_artifact_status="passed",
            )
            self.assertEqual(
                self.geo._run_with_runner(launch, dry_run=False, verbose=False), 1
            )
            self.assertFalse((root / "failure.json").exists())

    def test_forward_gate_launchers_are_single_isaac_components_or_direct_pipeline(self) -> None:
        parser = self.geo._build_parser()
        args, extra = parser.parse_known_args(
            ["walk", "train-forward-walk", "--num-envs", "64", "--iterations", "2"]
        )
        training = self.geo._build_spec(args, extra)
        self.assertEqual(training.runner, "isaac")
        self.assertEqual(training.argv[1:3], ["--mode", "train"])
        self.assertEqual(training.success_artifact.name, "training.json")

        args, extra = parser.parse_known_args(
            ["walk", "validate-forward-walk-dynamics", "--steps", "32", "--smoke"]
        )
        dynamics = self.geo._build_spec(args, extra)
        self.assertEqual(dynamics.runner, "isaac")
        self.assertEqual(dynamics.argv[1:3], ["--mode", "forward"])
        self.assertEqual(dynamics.success_artifact.name, "forward_dynamics_smoke_validation.json")

        args, extra = parser.parse_known_args(["walk", "validate-forward-walk", "--headless"])
        gate = self.geo._build_spec(args, extra)
        self.assertEqual(gate.runner, "direct")
        self.assertEqual(gate.success_artifact.name, "validation.json")

        args, extra = parser.parse_known_args([
            "walk", "probe-forward-reference", "--steps", "250", "--headless"
        ])
        probe = self.geo._build_spec(args, extra)
        self.assertEqual(probe.runner, "isaac")
        self.assertEqual(probe.argv[0], "algorithms/urdf_learn_wasd_walk/forward_reference_probe.py")
        self.assertEqual(probe.success_artifact.name, "reference_probe.json")
        self.assertEqual(probe.required_artifact_status, "passed")


if __name__ == "__main__":
    unittest.main()

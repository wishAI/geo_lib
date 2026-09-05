"""Run or finalize the two-process passive-stand evidence pipeline."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# ``geo walk finalize-passive`` intentionally executes this file by path.  In
# that mode Python adds this file's directory, not the repository root, to
# sys.path.  Bootstrap only the repository root before importing the package.
REPO_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_BOOTSTRAP_ROOT))

from algorithms.urdf_learn_wasd_walk import model_spec, passive_stand


REPO_ROOT = model_spec.ALGORITHM_ROOT.parent.parent
ISAACLAB_SH = Path("/home/wishai/vscode/IsaacLab/isaaclab.sh")
ISAAC_SCRIPT = model_spec.ALGORITHM_ROOT / "passive_stand.py"
MILESTONES_PATH = model_spec.ALGORITHM_ROOT / "milestones.json"
GUI_MANIFEST_PATH = model_spec.ALGORITHM_ROOT / "gui" / "manifest.json"


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _record_canonical_pass(
    final: dict,
    final_path: Path,
    dynamics_path: Path,
    proof_path: Path,
    video_path: Path,
    sheet_path: Path,
    *,
    milestones_path: Path = MILESTONES_PATH,
    manifest_path: Path = GUI_MANIFEST_PATH,
) -> None:
    """Replace invalidated gate-1 residue with the exact current-lineage pass."""

    milestones = json.loads(milestones_path.read_text(encoding="utf-8"))
    if milestones.get("lineage") != model_spec.LINEAGE:
        raise ValueError("canonical milestone ledger belongs to another lineage")
    if milestones.get("assetContract", {}).get("meshTreeSha256") != model_spec.EXPECTED_MESH_TREE_SHA256:
        raise ValueError("canonical milestone ledger names another mesh tree")
    records = {item["id"]: item for item in milestones.get("milestones", [])}
    current = records.get(passive_stand.MILESTONE_ID)
    following = records.get("stand_30s_no_reset")
    if current is None or current.get("status") not in {"in_progress", "passed"}:
        raise ValueError("passive stand is not the canonical active or passed milestone")
    if following is None or following.get("status") not in {"not_started", "in_progress"}:
        raise ValueError("policy stand has an incompatible canonical status")
    if final["input"].get("mesh_tree_sha256") != model_spec.EXPECTED_MESH_TREE_SHA256:
        raise ValueError("final passive evidence names another mesh tree")

    def declared(kind: str, path: Path) -> dict:
        return {
            "kind": kind,
            "path": str(path.resolve().relative_to(REPO_BOOTSTRAP_ROOT)),
            "sha256": passive_stand._sha256(path),
        }

    metrics = final["metrics"]
    current.clear()
    current.update({
        "order": 1,
        "id": passive_stand.MILESTONE_ID,
        "stage": "stand",
        "status": "passed",
        "passedAt": final["assembled_at"],
        "checkpoint": final["checkpoint"],
        "urdfSha256": final["input"]["urdf_sha256"],
        "meshTreeSha256": final["input"]["mesh_tree_sha256"],
        "seed": final["seed"],
        "metrics": {
            name: metrics[name] for name in (
                "duration_s", "reset_count", "done_count", "fall_count",
                "max_reference_tilt_rad", "root_height_drop_m", "horizontal_drift_m",
                "minimum_support_polygon_margin_m",
            )
        },
        "evidence": [
            declared("validation", final_path),
            declared("dynamics_validation", dynamics_path),
            declared("proof_validation", proof_path),
            declared("video", video_path),
            declared("contact_sheet", sheet_path),
        ],
    })
    # This record may still contain a checkpoint and evidence from the explicitly
    # invalidated mesh lineage.  Activating gate 2 must not leave those fields
    # looking resumable in the canonical current-lineage ledger.
    following.clear()
    following.update({
        "order": 2,
        "id": "stand_30s_no_reset",
        "stage": "stand",
        "status": "in_progress",
    })
    milestones["implementationStatus"] = "milestone_1_passed_milestone_2_in_progress"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_records = {item["id"]: item for item in manifest.get("milestones", [])}
    if set((passive_stand.MILESTONE_ID, "stand_30s_no_reset")) - set(manifest_records):
        raise ValueError("GUI manifest lacks the passive or policy stand milestone")
    manifest_records[passive_stand.MILESTONE_ID]["status"] = "passed"
    manifest_records["stand_30s_no_reset"]["status"] = "in progress"
    manifest["summary"] = (
        "Latest-mesh Landau passive standing is proven for 30 seconds; policy standing "
        "is now being rebuilt from the corrected visual/collision package."
    )
    manifest["runtimeLabel"] = "TK2 · latest-mesh policy stand"

    _write_json(milestones_path, milestones)
    _write_json(manifest_path, manifest)


def _load_component(path: Path, phase: str, smoke: bool) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {phase} component evidence: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != 3 or evidence.get("milestone") != passive_stand.MILESTONE_ID:
        raise ValueError(f"invalid {phase} component evidence schema or milestone")
    if evidence.get("component") != phase or evidence.get("scope") != "component_only":
        raise ValueError(f"evidence is not the {phase} component")
    if evidence.get("status") != "passed" or evidence.get("failures"):
        raise ValueError(f"{phase} component has not passed")
    if bool(evidence.get("gate_eligible")) == smoke:
        raise ValueError(f"{phase} component smoke/exact identity does not match")
    return evidence


def _artifact_path(output_dir: Path, relative_path: str, expected_name: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    if path.parent != output_dir or path.name != expected_name:
        raise ValueError(f"proof artifact escaped its exact output path: {path}")
    if not path.is_file():
        raise ValueError(f"proof artifact is missing: {path}")
    return path


def finalize(
    output_dir: Path,
    *,
    smoke: bool = False,
    milestones_path: Path = MILESTONES_PATH,
    manifest_path: Path = GUI_MANIFEST_PATH,
) -> dict:
    """Write final evidence only after both independent components pass."""
    output_dir = passive_stand.safe_output_dir(output_dir)
    final_path = output_dir / ("smoke_validation.json" if smoke else "validation.json")
    final_path.unlink(missing_ok=True)
    dynamics_path = output_dir / passive_stand.component_artifact_name("dynamics", smoke)
    proof_path = output_dir / passive_stand.component_artifact_name("proof", smoke)
    dynamics = _load_component(dynamics_path, "dynamics", smoke)
    proof = _load_component(proof_path, "proof", smoke)

    for key in ("lineage", "seed", "checkpoint", "input", "command_contract"):
        if dynamics[key] != proof[key]:
            raise ValueError(f"component mismatch for {key}")
    if dynamics["joint_contract"]["action_joints"] != proof["joint_contract"]["action_joints"]:
        raise ValueError("component action-joint contracts differ")
    if dynamics["joint_contract"]["locked_joints"] != proof["joint_contract"]["locked_joints"]:
        raise ValueError("component locked-joint contracts differ")

    required = 0.0 if smoke else passive_stand.MIN_GATE_DURATION_S
    for label, component in (("dynamics", dynamics), ("proof", proof)):
        passed, failures = passive_stand.evaluate_gate(component["metrics"], required_duration_s=required)
        if not passed:
            raise ValueError(f"{label} metrics failed revalidation: {failures}")
        support_failures = passive_stand.evaluate_free_root_support(component["metrics"])
        if support_failures:
            raise ValueError(f"{label} support metrics failed revalidation: {support_failures}")
    imported = dynamics["joint_contract"].get("runtime_importer_axis_evidence")
    if not imported or not imported.get("passed") or imported.get("joint_count") != 69:
        raise ValueError("dynamics evidence lacks a passing 69-joint importer-axis audit")

    video = proof.get("video_inspection")
    if not video:
        raise ValueError("proof component lacks video inspection")
    passed, failures = passive_stand.evaluate_proof(video, required_duration_s=required)
    if not passed:
        raise ValueError(f"proof inspection failed revalidation: {failures}")
    expected_video = "proof_smoke.mp4" if smoke else "proof.mp4"
    expected_sheet = "contact_sheet_smoke.png" if smoke else "contact_sheet.png"
    video_path = _artifact_path(output_dir, video["path"], expected_video)
    sheet_path = _artifact_path(output_dir, video["contact_sheet_path"], expected_sheet)
    if passive_stand._sha256(video_path) != video["sha256"]:
        raise ValueError("proof video hash does not match proof evidence")
    if passive_stand._sha256(sheet_path) != video["contact_sheet_sha256"]:
        raise ValueError("contact-sheet hash does not match proof evidence")

    status = "smoke_passed_not_promotable" if smoke else "passed"
    final = {
        "schema_version": 3,
        "milestone": passive_stand.MILESTONE_ID,
        "status": status,
        "lineage": dynamics["lineage"],
        "assembled_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "seed": dynamics["seed"],
        "checkpoint": dynamics["checkpoint"],
        "versions": {"dynamics": dynamics["versions"], "proof": proof["versions"]},
        "simulator": {
            "single_process_per_component": True,
            "components_ran_sequentially": True,
            "dynamics_rendering_enabled": False,
            "proof_camera_sensor_created": False,
            "proof_capture_path": video["capture_path"],
        },
        "input": dynamics["input"],
        "joint_contract": dynamics["joint_contract"],
        "command_contract": dynamics["command_contract"],
        "metrics": dynamics["metrics"],
        "proof_metrics": proof["metrics"],
        "traces": dynamics["traces"],
        "proof_traces": proof["traces"],
        "video_inspection": video,
        "component_evidence": {
            "dynamics": {
                "path": str(dynamics_path.relative_to(REPO_ROOT)),
                "sha256": passive_stand._sha256(dynamics_path),
                "run_identity": dynamics["run_identity"],
            },
            "proof": {
                "path": str(proof_path.relative_to(REPO_ROOT)),
                "sha256": passive_stand._sha256(proof_path),
                "run_identity": proof["run_identity"],
            },
        },
        "failures": [],
        "cumulative_gates": [{"order": 1, "id": passive_stand.MILESTONE_ID, "status": status}],
    }
    final_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not smoke:
        try:
            _record_canonical_pass(
                final, final_path, dynamics_path, proof_path, video_path, sheet_path,
                milestones_path=milestones_path, manifest_path=manifest_path,
            )
        except Exception:
            final_path.unlink(missing_ok=True)
            raise
    print(json.dumps({
        "status": status,
        "milestone": passive_stand.MILESTONE_ID,
        "validation": os.fspath(final_path),
        "proof_video": os.fspath(video_path),
        "contact_sheet": os.fspath(sheet_path),
    }, indent=2))
    return final


def _run_component(args, phase: str) -> None:
    argv = [
        os.fspath(ISAACLAB_SH), "-p", os.fspath(ISAAC_SCRIPT), "--phase", phase,
        "--output-dir", os.fspath(args.output_dir), "--duration", str(args.duration),
        "--seed", str(args.seed), "--video-fps", str(args.video_fps),
        "--video-width", str(args.video_width), "--video-height", str(args.video_height),
        "--device", args.device,
    ]
    if args.headless:
        argv.append("--headless")
    if args.steps is not None:
        argv.extend(["--steps", str(args.steps)])
    if args.smoke:
        argv.append("--smoke")
    env = os.environ.copy()
    env["TERM"] = "xterm"
    completed = subprocess.run(argv, cwd=REPO_ROOT, env=env, check=False)
    artifact = args.output_dir / passive_stand.component_artifact_name(phase, args.smoke)
    if completed.returncode != 0:
        raise RuntimeError(f"{phase} component exited {completed.returncode}; retained evidence: {artifact}")
    if not artifact.is_file():
        raise RuntimeError(f"{phase} component exited zero without its success artifact: {artifact}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=passive_stand.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration", type=float, default=passive_stand.MIN_GATE_DURATION_S)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir = passive_stand.safe_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / ("smoke_validation.json" if args.smoke else "validation.json")
    if args.finalize_only:
        try:
            finalize(args.output_dir, smoke=args.smoke)
            return 0
        except ValueError as error:
            print(f"Passive evidence finalization failed: {error}", file=sys.stderr)
            return 1

    if not ISAACLAB_SH.is_file():
        raise SystemExit(f"Isaac Lab launcher not found: {ISAACLAB_SH}")
    lock_path = args.output_dir / ".passive_pipeline.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another passive validation pipeline holds the single-process lock.", file=sys.stderr)
            return 1
        for name in (
            final_path.name,
            passive_stand.component_artifact_name("dynamics", args.smoke),
            passive_stand.component_artifact_name("proof", args.smoke),
            "proof_smoke.mp4" if args.smoke else "proof.mp4",
            "contact_sheet_smoke.png" if args.smoke else "contact_sheet.png",
        ):
            (args.output_dir / name).unlink(missing_ok=True)
        try:
            _run_component(args, "dynamics")
            _run_component(args, "proof")
            finalize(args.output_dir, smoke=args.smoke)
        except (RuntimeError, ValueError) as error:
            final_path.unlink(missing_ok=True)
            print(f"Passive validation stopped fail-closed: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

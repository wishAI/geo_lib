"""Run or finalize sequential policy-stand dynamics and viewport-proof evidence."""

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

REPO_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_BOOTSTRAP_ROOT))

from algorithms.urdf_learn_wasd_walk import model_spec, passive_stand
from algorithms.urdf_learn_wasd_walk import policy_stand_contract as contract


ISAACLAB_SH = Path("/home/wishai/vscode/IsaacLab/isaaclab.sh")
ISAAC_SCRIPT = model_spec.ALGORITHM_ROOT / "policy_stand.py"
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
    training_path: Path,
    *,
    milestones_path: Path = MILESTONES_PATH,
    manifest_path: Path = GUI_MANIFEST_PATH,
) -> None:
    """Idempotently reconcile exact gate evidence into both canonical status files."""

    milestones = json.loads(milestones_path.read_text(encoding="utf-8"))
    if milestones.get("lineage") != contract.LINEAGE:
        raise ValueError("canonical milestone ledger belongs to another lineage")
    records = {item["id"]: item for item in milestones.get("milestones", [])}
    prior = records.get(contract.PRIOR_MILESTONE_ID)
    current = records.get(contract.MILESTONE_ID)
    following = records.get("gate_5m_no_reset")
    if prior is None or prior.get("status") != "passed":
        raise ValueError("canonical ledger no longer records the passive gate as passed")
    if current is None or current.get("status") not in {"in_progress", "passed"}:
        raise ValueError("policy stand is not the canonical active or passed milestone")
    if following is None or following.get("status") not in {"not_started", "in_progress"}:
        raise ValueError("5 m gate has an incompatible canonical status")

    def declared(kind: str, path: Path) -> dict:
        return {
            "kind": kind,
            "path": str(path.resolve().relative_to(REPO_BOOTSTRAP_ROOT)),
            "sha256": contract.sha256(path),
        }

    metrics = final["metrics"]
    current.clear()
    current.update({
        "order": 2,
        "id": contract.MILESTONE_ID,
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
            declared("training", training_path),
            declared("dynamics_validation", dynamics_path),
            declared("proof_validation", proof_path),
            declared("checkpoint", REPO_BOOTSTRAP_ROOT / final["checkpoint"]["path"]),
            declared("video", video_path),
            declared("contact_sheet", sheet_path),
        ],
    })
    # Never leave the invalidated mesh lineage's status/checkpoint residue on
    # the newly activated forward gate.
    following.clear()
    following.update({
        "order": 3,
        "id": "gate_5m_no_reset",
        "stage": "fwd_only",
        "status": "in_progress",
    })
    milestones["implementationStatus"] = "milestone_2_passed_milestone_3_in_progress"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_records = {item["id"]: item for item in manifest.get("milestones", [])}
    if set((contract.MILESTONE_ID, "gate_5m_no_reset")) - set(manifest_records):
        raise ValueError("GUI manifest lacks the policy or 5 m milestone")
    manifest_records[contract.MILESTONE_ID]["status"] = "passed"
    manifest_records["gate_5m_no_reset"]["status"] = "in progress"
    manifest["summary"] = (
        "Clean-lineage Landau locomotion rebuilt one hard gate at a time. Passive and "
        "policy standing are proven; the 5 m flat forward gate is active."
    )
    manifest["runtimeLabel"] = "TK2 · 5 m forward gate"
    manifest["capabilities"] = [
        "passive stand", "policy stand", "forward walk", "Isaac Lab", "clean room"
    ]

    _write_json(milestones_path, milestones)
    _write_json(manifest_path, manifest)


def _load_component(path: Path, phase: str, smoke: bool) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {phase} component evidence: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != 1 or evidence.get("milestone") != contract.MILESTONE_ID:
        raise ValueError(f"invalid {phase} component schema or milestone")
    if evidence.get("component") != phase or evidence.get("scope") != "component_only":
        raise ValueError(f"evidence is not the {phase} policy component")
    if evidence.get("status") != "passed" or evidence.get("failures"):
        raise ValueError(f"{phase} policy component has not passed")
    if bool(evidence.get("gate_eligible")) == smoke:
        raise ValueError(f"{phase} policy component smoke/exact identity differs")
    return evidence


def _exact_artifact(output_dir: Path, relative_path: str, expected_name: str) -> Path:
    path = (REPO_BOOTSTRAP_ROOT / relative_path).resolve()
    if path.parent != output_dir or path.name != expected_name or not path.is_file():
        raise ValueError(f"proof artifact is absent or escaped its exact output path: {path}")
    return path


def finalize(output_dir: Path, *, smoke: bool = False) -> dict:
    output_dir = contract.safe_output_dir(output_dir)
    prior = contract.load_prior_gate()
    training = contract.load_training_evidence(output_dir)
    dynamics_path = output_dir / contract.component_artifact_name("dynamics", smoke)
    proof_path = output_dir / contract.component_artifact_name("proof", smoke)
    dynamics = _load_component(dynamics_path, "dynamics", smoke)
    proof = _load_component(proof_path, "proof", smoke)
    for key in ("lineage", "seed", "checkpoint", "training_evidence", "input", "policy_contract"):
        if dynamics[key] != proof[key]:
            raise ValueError(f"policy components differ for {key}")
    if dynamics["checkpoint"]["sha256"] != training["checkpoint"]["sha256"]:
        raise ValueError("components do not evaluate the current training checkpoint")
    training_path = output_dir / contract.TRAINING_EVIDENCE
    if dynamics["training_evidence"] != {
        "path": str(training_path.relative_to(REPO_BOOTSTRAP_ROOT)),
        "sha256": contract.sha256(training_path),
        "run_identity": training["run_identity"],
    }:
        raise ValueError("components do not identify the current training evidence")
    for component in (dynamics, proof):
        if component.get("cumulative_gates", [None])[0] != prior:
            raise ValueError("policy component does not carry the current exact passive gate")
    for component_name, component in (("dynamics", dynamics), ("proof", proof)):
        required = 0.0 if smoke else contract.MIN_GATE_DURATION_S
        failures = contract.evaluate_policy_gate(component["metrics"], required_duration_s=required)
        failures.extend(passive_stand.evaluate_free_root_support(component["metrics"]))
        if failures:
            raise ValueError(f"{component_name} metrics failed revalidation: {failures}")
    imported = dynamics["joint_contract"].get("runtime_importer_axis_evidence")
    if not imported or not imported.get("passed") or imported.get("joint_count") != 69:
        raise ValueError("camera-free policy evidence lacks a passing 69-joint importer-axis audit")
    if dynamics["joint_contract"]["runtime_action_order"] != list(model_spec.ACTION_JOINTS):
        raise ValueError("camera-free policy evidence has the wrong action order")

    video = proof.get("video_inspection")
    if not video:
        raise ValueError("policy proof component lacks video inspection")
    required = 0.0 if smoke else contract.MIN_GATE_DURATION_S
    passed, failures = passive_stand.evaluate_proof(video, required_duration_s=required)
    if not passed:
        raise ValueError(f"policy proof inspection failed revalidation: {failures}")
    expected_video = "proof_smoke.mp4" if smoke else "proof.mp4"
    expected_sheet = "contact_sheet_smoke.png" if smoke else "contact_sheet.png"
    video_path = _exact_artifact(output_dir, video["path"], expected_video)
    sheet_path = _exact_artifact(output_dir, video["contact_sheet_path"], expected_sheet)
    if contract.sha256(video_path) != video["sha256"]:
        raise ValueError("policy proof video hash differs from proof evidence")
    if contract.sha256(sheet_path) != video["contact_sheet_sha256"]:
        raise ValueError("policy contact-sheet hash differs from proof evidence")

    status = "smoke_passed_not_promotable" if smoke else "passed"
    final_path = output_dir / ("smoke_validation.json" if smoke else "validation.json")
    canonical = not smoke and output_dir == contract.DEFAULT_OUTPUT_DIR.resolve()
    final = {
        "schema_version": 1,
        "milestone": contract.MILESTONE_ID,
        "status": status,
        "lineage": contract.LINEAGE,
        "assembled_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "seed": dynamics["seed"],
        "checkpoint": dynamics["checkpoint"],
        "training_evidence": dynamics["training_evidence"],
        "source_commit": dynamics["source_commit"],
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
        "policy_contract": dynamics["policy_contract"],
        "command_contract": dynamics["command_contract"],
        "metrics": dynamics["metrics"],
        "proof_metrics": proof["metrics"],
        "traces": dynamics["traces"],
        "action_traces": dynamics["action_traces"],
        "proof_traces": proof["traces"],
        "proof_action_traces": proof["action_traces"],
        "video_inspection": video,
        "component_evidence": {
            "dynamics": {
                "path": str(dynamics_path.relative_to(REPO_BOOTSTRAP_ROOT)),
                "sha256": contract.sha256(dynamics_path),
                "run_identity": dynamics["run_identity"],
            },
            "proof": {
                "path": str(proof_path.relative_to(REPO_BOOTSTRAP_ROOT)),
                "sha256": contract.sha256(proof_path),
                "run_identity": proof["run_identity"],
            },
        },
        "milestone_status_changed": canonical,
        "failures": [],
        "cumulative_gates": [prior, {"order": 2, "id": contract.MILESTONE_ID, "status": status}],
    }
    _write_json(final_path, final)
    if canonical:
        _record_canonical_pass(
            final, final_path, dynamics_path, proof_path, video_path, sheet_path, training_path
        )
    print(json.dumps({
        "status": status, "milestone": contract.MILESTONE_ID,
        "validation": os.fspath(final_path), "checkpoint": final["checkpoint"],
        "proof_video": os.fspath(video_path), "contact_sheet": os.fspath(sheet_path),
    }, indent=2))
    return final


def _run_component(args, phase: str) -> None:
    argv = [
        os.fspath(ISAACLAB_SH), "-p", os.fspath(ISAAC_SCRIPT), "--mode", phase,
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
    if args.reuse_usd_cache:
        argv.append("--reuse-usd-cache")
    completed = subprocess.run(argv, cwd=REPO_BOOTSTRAP_ROOT, env={**os.environ, "TERM": "xterm"}, check=False)
    artifact = args.output_dir / contract.component_artifact_name(phase, args.smoke)
    if completed.returncode != 0:
        raise RuntimeError(f"{phase} policy component exited {completed.returncode}; retained evidence: {artifact}")
    if not artifact.is_file():
        raise RuntimeError(f"{phase} policy component exited zero without evidence: {artifact}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=contract.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration", type=float, default=contract.MIN_GATE_DURATION_S)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-usd-cache", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir = contract.safe_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / ("smoke_validation.json" if args.smoke else "validation.json")
    if args.finalize_only:
        try:
            finalize(args.output_dir, smoke=args.smoke)
            return 0
        except ValueError as error:
            final_path.unlink(missing_ok=True)
            print(f"Policy evidence finalization failed: {error}", file=sys.stderr)
            return 1
    if args.reuse_usd_cache and not args.smoke:
        raise SystemExit("--reuse-usd-cache is restricted to non-promotable pipeline smoke runs")
    if not ISAACLAB_SH.is_file():
        raise SystemExit(f"Isaac Lab launcher not found: {ISAACLAB_SH}")
    lock_path = args.output_dir / ".policy_stand_pipeline.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another policy stand pipeline holds the single-process lock.", file=sys.stderr)
            return 1
        for name in (
            final_path.name,
            contract.component_artifact_name("dynamics", args.smoke),
            contract.component_artifact_name("proof", args.smoke),
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
            print(f"Policy validation stopped fail-closed: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sequential fail-closed assembly for Landau's cumulative 5 m gate."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.urdf_learn_wasd_walk import forward_walk_contract as contract
from algorithms.urdf_learn_wasd_walk import model_spec, passive_stand


ISAACLAB_SH = Path("/home/wishai/vscode/IsaacLab/isaaclab.sh")
ISAAC_SCRIPT = model_spec.ALGORITHM_ROOT / "forward_walk.py"
MILESTONES_PATH = model_spec.ALGORITHM_ROOT / "milestones.json"
GUI_MANIFEST_PATH = model_spec.ALGORITHM_ROOT / "gui" / "manifest.json"


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _record_canonical_pass(
    final: dict,
    final_path: Path,
    output: Path,
    video_path: Path,
    sheet_path: Path,
    *,
    milestones_path: Path = MILESTONES_PATH,
    manifest_path: Path = GUI_MANIFEST_PATH,
) -> None:
    ledger = json.loads(milestones_path.read_text(encoding="utf-8"))
    records = {item["id"]: item for item in ledger.get("milestones", [])}
    current = records.get(contract.MILESTONE_ID)
    following = records.get("gate_10m_no_reset")
    if ledger.get("lineage") != contract.LINEAGE:
        raise ValueError("canonical milestone ledger belongs to another lineage")
    if current is None or current.get("status") not in {"in_progress", "passed"}:
        raise ValueError("5 m gate is not the canonical active or passed milestone")
    if following is None or following.get("status") not in {"not_started", "in_progress"}:
        raise ValueError("10 m gate has an incompatible canonical status")

    def declared(kind: str, path: Path) -> dict:
        return {
            "kind": kind,
            "path": str(path.resolve().relative_to(REPO_ROOT)),
            "sha256": contract.sha256(path),
        }

    current.clear()
    current.update({
        "order": 3,
        "id": contract.MILESTONE_ID,
        "stage": "fwd_only",
        "status": "passed",
        "passedAt": final["assembled_at"],
        "checkpoint": final["checkpoint"],
        "urdfSha256": final["input"]["urdf_sha256"],
        "seed": final["seed"],
        "metrics": final["metrics"],
        "candidateCheckpointStandRepass": final["cumulative_gate_metrics"]["stand_30s_no_reset"],
        "evidence": [
            declared("validation", final_path),
            declared("training", output / contract.TRAINING_EVIDENCE),
            declared("stand_dynamics_validation", output / contract.component_artifact_name("stand")),
            declared("forward_dynamics_validation", output / contract.component_artifact_name("forward")),
            declared("proof_validation", output / contract.component_artifact_name("proof")),
            declared("checkpoint", REPO_ROOT / final["checkpoint"]["path"]),
            declared("video", video_path),
            declared("contact_sheet", sheet_path),
        ],
    })
    following["status"] = "in_progress"
    ledger["implementationStatus"] = "milestone_3_passed_milestone_4_in_progress"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gui = {item["id"]: item for item in manifest.get("milestones", [])}
    if set((contract.MILESTONE_ID, "gate_10m_no_reset")) - set(gui):
        raise ValueError("GUI manifest lacks the 5 m or 10 m milestone")
    gui[contract.MILESTONE_ID]["status"] = "passed"
    gui["gate_10m_no_reset"]["status"] = "in progress"
    manifest["summary"] = (
        "Clean-lineage Landau locomotion rebuilt one hard gate at a time. The 5 m flat "
        "forward gate is proven; the cumulative 10 m gate is active."
    )
    manifest["runtimeLabel"] = "TK2 · 10 m forward gate"
    _write_json(milestones_path, ledger)
    _write_json(manifest_path, manifest)


def _load_component(path: Path, mode: str, smoke: bool) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {mode} component evidence: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (
        evidence.get("schema_version") != 1
        or evidence.get("milestone") != contract.MILESTONE_ID
        or evidence.get("component") != mode
        or evidence.get("scope") != "component_only"
    ):
        raise ValueError(f"invalid {mode} component identity")
    if evidence.get("status") != "passed" or evidence.get("failures"):
        raise ValueError(f"{mode} component has not passed")
    if bool(evidence.get("gate_eligible")) == smoke:
        raise ValueError(f"{mode} smoke/exact identity differs")
    return evidence


def _exact_artifact(output: Path, relative: str, name: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    if path.parent != output or path.name != name or not path.is_file():
        raise ValueError(f"proof artifact is absent or escaped its exact path: {path}")
    return path


def finalize(output_dir: Path, *, smoke: bool = False) -> dict:
    output = contract.safe_output_dir(output_dir)
    prior, _ = contract.load_cumulative_prior()
    training = contract.load_training_evidence(output)
    components = {
        mode: _load_component(
            output / contract.component_artifact_name(mode, smoke), mode, smoke
        )
        for mode in ("stand", "forward", "proof")
    }
    stand, forward, proof = (components[name] for name in ("stand", "forward", "proof"))
    for key in ("lineage", "seed", "checkpoint", "training_evidence", "input", "policy_contract"):
        if stand[key] != forward[key] or forward[key] != proof[key]:
            raise ValueError(f"5 m components differ for {key}")
    if stand["checkpoint"]["sha256"] != training["checkpoint"]["sha256"]:
        raise ValueError("components do not evaluate the current forward checkpoint")
    if stand["cumulative_gates"][:2] != prior:
        raise ValueError("candidate stand replay does not carry exact prior gates")
    for component in (stand, forward):
        imported = component["joint_contract"].get("runtime_importer_axis_evidence")
        if not imported or not imported.get("passed") or imported.get("joint_count") != 69:
            raise ValueError(f"{component['component']} lacks a passing importer-axis audit")
    stand_failures = contract.evaluate_repassed_stand(
        stand["metrics"], required_duration_s=0.0 if smoke else contract.STAND_DURATION_S
    )
    required_distance = 0.0 if smoke else contract.TARGET_DISTANCE_M
    forward_failures = contract.evaluate_forward_gate(
        forward["metrics"], required_distance_m=required_distance, require_gait=not smoke
    )
    proof_failures = contract.evaluate_forward_gate(
        proof["metrics"], required_distance_m=required_distance, require_gait=not smoke
    )
    if stand_failures or forward_failures or proof_failures:
        raise ValueError(
            f"component metrics failed revalidation: stand={stand_failures}, "
            f"forward={forward_failures}, proof={proof_failures}"
        )
    video = proof.get("video_inspection")
    if not video or not video.get("motion_discernible"):
        raise ValueError("forward proof lacks discernible full-gate motion")
    proof_passed, video_failures = passive_stand.evaluate_proof(
        video, required_duration_s=max(0.0, proof["metrics"]["duration_s"] - 0.25)
    )
    if not proof_passed:
        raise ValueError(f"forward proof inspection failed: {video_failures}")
    video_path = _exact_artifact(
        output, video["path"], "proof_smoke.mp4" if smoke else "proof.mp4"
    )
    sheet_path = _exact_artifact(
        output,
        video["contact_sheet_path"],
        "contact_sheet_smoke.png" if smoke else "contact_sheet.png",
    )
    if contract.sha256(video_path) != video["sha256"]:
        raise ValueError("forward proof video hash differs")
    if contract.sha256(sheet_path) != video["contact_sheet_sha256"]:
        raise ValueError("forward contact-sheet hash differs")

    status = "smoke_passed_not_promotable" if smoke else "passed"
    final_path = output / ("smoke_validation.json" if smoke else "validation.json")
    canonical = not smoke and output == contract.DEFAULT_OUTPUT_DIR.resolve()
    final = {
        "schema_version": 1,
        "milestone": contract.MILESTONE_ID,
        "status": status,
        "lineage": contract.LINEAGE,
        "assembled_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "seed": stand["seed"],
        "checkpoint": stand["checkpoint"],
        "training_evidence": stand["training_evidence"],
        "source_commit": stand["source_commit"],
        "versions": {name: component["versions"] for name, component in components.items()},
        "simulator": {
            "single_process_per_component": True,
            "components_ran_sequentially": True,
            "camera_sensor_created": False,
            "physics_components": ["candidate_policy_stand_30s", "forward_5m"],
        },
        "input": stand["input"],
        "joint_contract": stand["joint_contract"],
        "policy_contract": stand["policy_contract"],
        "command_contract": {
            "stand": stand["command_contract"],
            "forward": forward["command_contract"],
        },
        "cumulative_gate_metrics": {
            "stand_30s_no_reset": stand["metrics"],
            contract.MILESTONE_ID: forward["metrics"],
        },
        "metrics": forward["metrics"],
        "traces": forward["traces"],
        "action_traces": forward["action_traces"],
        "stand_repass_traces": stand["traces"],
        "video_inspection": video,
        "component_evidence": {
            name: {
                "path": str((output / contract.component_artifact_name(name, smoke)).relative_to(REPO_ROOT)),
                "sha256": contract.sha256(output / contract.component_artifact_name(name, smoke)),
                "run_identity": component["run_identity"],
            }
            for name, component in components.items()
        },
        "milestone_status_changed": canonical,
        "failures": [],
        "cumulative_gates": [
            prior[0],
            {
                **prior[1],
                "candidate_checkpoint_repassed": True,
                "candidate_checkpoint_sha256": stand["checkpoint"]["sha256"],
                "repass_component": str(
                    (output / contract.component_artifact_name("stand", smoke)).relative_to(REPO_ROOT)
                ),
            },
            {"order": 3, "id": contract.MILESTONE_ID, "status": status},
        ],
    }
    _write_json(final_path, final)
    if canonical:
        _record_canonical_pass(final, final_path, output, video_path, sheet_path)
    print(json.dumps({
        "status": status,
        "milestone": contract.MILESTONE_ID,
        "validation": str(final_path),
        "checkpoint": final["checkpoint"],
        "forward_metrics": forward["metrics"],
        "proof_video": str(video_path),
    }, indent=2))
    return final


def _run_component(args, mode: str) -> None:
    argv = [
        os.fspath(ISAACLAB_SH), "-p", os.fspath(ISAAC_SCRIPT),
        "--mode", mode, "--output-dir", os.fspath(args.output_dir),
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
    if mode == "proof" or args.reuse_usd_cache:
        argv.append("--reuse-usd-cache")
    completed = subprocess.run(
        argv, cwd=REPO_ROOT, env={**os.environ, "TERM": "xterm"}, check=False
    )
    artifact = args.output_dir / contract.component_artifact_name(mode, args.smoke)
    if completed.returncode != 0:
        raise RuntimeError(f"{mode} component exited {completed.returncode}; evidence: {artifact}")
    if not artifact.is_file():
        raise RuntimeError(f"{mode} component exited zero without evidence: {artifact}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=contract.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
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
            print(f"Forward finalization failed: {error}", file=sys.stderr)
            return 1
    lock_path = args.output_dir / ".forward_walk_pipeline.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another forward pipeline holds the single-process lock.", file=sys.stderr)
            return 1
        for name in (
            final_path.name,
            *(contract.component_artifact_name(mode, args.smoke) for mode in ("stand", "forward", "proof")),
            "proof_smoke.mp4" if args.smoke else "proof.mp4",
            "contact_sheet_smoke.png" if args.smoke else "contact_sheet.png",
        ):
            (args.output_dir / name).unlink(missing_ok=True)
        try:
            for mode in ("stand", "forward", "proof"):
                _run_component(args, mode)
            finalize(args.output_dir, smoke=args.smoke)
        except (RuntimeError, ValueError) as error:
            final_path.unlink(missing_ok=True)
            print(f"Forward validation stopped fail-closed: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

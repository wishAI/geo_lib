from __future__ import annotations

from pathlib import Path
from typing import Any

from .run_history import resolve_recommended_checkpoint
from .training_lineage import load_active_lineage, resolve_landau_experiment_name


def _is_default_checkpoint_selector(load_run: Any, checkpoint: Any) -> bool:
    normalized_run = None if load_run is None else str(load_run).strip()
    normalized_checkpoint = None if checkpoint is None else str(checkpoint).strip()
    return normalized_run in {None, "", ".*"} and normalized_checkpoint in {None, "", "model_.*.pt"}


def preflight_default_checkpoint_selection(args: Any, *, mode: str) -> None:
    if getattr(args, "robot", None) != "landau":
        return
    if bool(getattr(args, "latest", False)):
        return
    load_run = getattr(args, "load_run", None)
    checkpoint = getattr(args, "checkpoint", None)
    if not _is_default_checkpoint_selector(load_run, checkpoint):
        return

    stage = getattr(args, "stage", None) or "full"
    experiment_name = getattr(args, "experiment_name", None) or resolve_landau_experiment_name(stage)
    if resolve_recommended_checkpoint(experiment_name) is not None:
        return

    active_lineage = load_active_lineage()
    if isinstance(active_lineage, dict):
        current_stage = active_lineage.get("current_stage")
        current_run = active_lineage.get("current_run")
        current_checkpoint = active_lineage.get("current_checkpoint")
        experiments = active_lineage.get("experiments")
        if (
            isinstance(current_stage, str)
            and current_stage
            and isinstance(current_run, str)
            and current_run
            and isinstance(current_checkpoint, str)
            and current_checkpoint
            and current_stage == stage
        ):
            checkpoint_name = Path(current_checkpoint).name
            if checkpoint_name.startswith("model_") and checkpoint_name.endswith(".pt"):
                args.load_run = current_run
                args.checkpoint = checkpoint_name
                if isinstance(experiments, dict):
                    resolved_experiment_name = experiments.get(current_stage)
                    if isinstance(resolved_experiment_name, str) and resolved_experiment_name:
                        args.experiment_name = resolved_experiment_name
                print(
                    "[INFO] No promoted checkpoint for "
                    f"'{experiment_name}'; using the active lineage current checkpoint.",
                    flush=True,
                )
                print(
                    f"[INFO] Default {mode} checkpoint: run={current_run} checkpoint={checkpoint_name}",
                    flush=True,
                )
                return

    message_lines = [
        f"No promoted checkpoint is registered for '{experiment_name}'.",
        f"Refusing to start Isaac for `walk {mode}` because no current fallback checkpoint exists either.",
    ]
    if isinstance(active_lineage, dict):
        lineage_name = active_lineage.get("lineage_name")
        current_stage = active_lineage.get("current_stage")
        current_run = active_lineage.get("current_run")
        current_checkpoint = active_lineage.get("current_checkpoint")
        if isinstance(lineage_name, str) and lineage_name:
            message_lines.append(f"Active lineage: {lineage_name}")
        if isinstance(current_stage, str) and current_stage:
            message_lines.append(f"Current stage: {current_stage}")
            if current_stage != stage:
                message_lines.append(
                    f"Requested stage '{stage}' does not match the active lineage stage '{current_stage}'."
                )
        if isinstance(current_run, str) and current_run:
            message_lines.append(f"Current run: {current_run}")
        if isinstance(current_checkpoint, str) and current_checkpoint:
            message_lines.append(f"Current checkpoint: {current_checkpoint}")
    message_lines.append(
        "Pass `--latest` to use the newest checkpoint from this experiment, or pass "
        "`--load_run <run> --checkpoint model_<N>.pt` explicitly."
    )
    raise SystemExit("\n".join(message_lines))

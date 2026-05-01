from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path


_DEFAULT_EXCLUDED_EXTENSIONS = (
    "omni.kit.telemetry",
    "omni.graph.telemetry",
    "omni.physx.telemetry",
    "omni.physx.clashdetection.telemetry",
    "isaacsim.robot.wheeled_robots",
)


def _prepare_project_documents_dir() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    omni_documents = repo_root / ".isaac_documents" / "Kit"
    shared_documents = omni_documents / "shared"
    app_documents = omni_documents / "apps" / "Isaac-Sim"
    (shared_documents / "screenshots").mkdir(parents=True, exist_ok=True)
    (app_documents / "scripts" / "new_stage").mkdir(parents=True, exist_ok=True)
    if not os.environ.get("XDG_DOCUMENTS_DIR"):
        os.environ["XDG_DOCUMENTS_DIR"] = str(repo_root / ".isaac_documents")
    return {
        "omni_documents": omni_documents,
        "shared_documents": shared_documents,
        "app_documents": app_documents,
    }


def apply_project_kit_args(args: Namespace) -> Namespace:
    document_tokens = _prepare_project_documents_dir()

    existing = str(getattr(args, "kit_args", "") or "").strip()
    extra_parts: list[str] = []
    for token_name, token_path in document_tokens.items():
        token = f"--/app/tokens/{token_name}={token_path}"
        if token not in existing:
            extra_parts.append(token)
    for index, ext_name in enumerate(_DEFAULT_EXCLUDED_EXTENSIONS):
        token = f"--/app/extensions/excluded/{index}={ext_name}"
        if token not in existing:
            extra_parts.append(token)
    if extra_parts:
        args.kit_args = " ".join(part for part in (existing, *extra_parts) if part).strip()
    return args

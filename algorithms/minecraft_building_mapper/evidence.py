"""Resolution labels and evidence validation for block catalog entries."""

from __future__ import annotations

from typing import Any, Iterable


RESOLUTION_KINDS = ("exact", "resolved", "inferred", "unknown")


def resolution_entry(
    resolution: str,
    *,
    name: str,
    provenance: Iterable[dict[str, Any]],
    reason: str,
    confidence: float | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Create an entry whose certainty can be audited without reading prose docs.

    ``exact`` means the supplied runtime files directly prove ID, metadata, and
    pixels. ``resolved`` combines an exact supplied asset with a stable external
    format mapping (for example a canonical vanilla numeric ID). ``inferred`` is
    always explicit and must be less than fully confident. ``unknown`` is never
    renderable.
    """

    if resolution not in RESOLUTION_KINDS:
        raise ValueError(f"Unsupported resolution: {resolution}")
    evidence = list(provenance)
    if not evidence or any(not isinstance(item, dict) for item in evidence):
        raise ValueError("provenance must contain at least one machine-readable evidence object")
    if not reason.strip():
        raise ValueError("every resolution requires a non-empty reason")
    if resolution == "inferred":
        if confidence is None or not 0.0 < confidence < 1.0:
            raise ValueError("inferred entries require confidence strictly between 0 and 1")
    elif resolution == "unknown":
        if confidence not in (None, 0, 0.0):
            raise ValueError("unknown entries cannot claim positive confidence")
        confidence = 0.0
    else:
        if confidence not in (None, 1, 1.0):
            raise ValueError(f"{resolution} entries require confidence 1.0")
        confidence = 1.0
    status = "unknown" if resolution == "unknown" else "resolved"
    return {
        "status": status,
        "resolution": resolution,
        "confidence": confidence,
        "name": name,
        "reason": reason,
        "provenance": evidence,
        **fields,
    }


def validate_resolution_entry(entry: dict[str, Any]) -> None:
    """Reject missing or misleading inference/provenance labels."""

    rebuilt = resolution_entry(
        str(entry.get("resolution", "")),
        name=str(entry.get("name", "")),
        provenance=entry.get("provenance", []),
        reason=str(entry.get("reason", "")),
        confidence=entry.get("confidence"),
    )
    if entry.get("status") != rebuilt["status"]:
        raise ValueError("catalog status conflicts with resolution label")

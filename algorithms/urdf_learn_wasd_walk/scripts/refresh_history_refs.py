from __future__ import annotations

from algorithms.urdf_learn_wasd_walk.run_history import refresh_history_refs


def main() -> None:
    payload = refresh_history_refs()
    print(
        f"[HISTORY] refreshed refs for {len(payload.get('experiments', {}))} experiment(s) "
        f"under {payload.get('history_root')}",
        flush=True,
    )


if __name__ == "__main__":
    main()

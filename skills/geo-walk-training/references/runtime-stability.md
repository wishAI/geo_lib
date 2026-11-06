# TK2 long-run and GUI stability

Use this reference for this project's unattended training and cloud-backed GUI. It records observed failure modes and decision points; it does not establish that resource limits alone cure a kernel or driver defect.

## Keep long runs bounded and resumable

- Use the persistent managed Codex backend. An SSH client is a detachable controller, not the lifetime owner of a training worker.
- Before a long batch, check host memory, free workspace space, current worker count, and basic SSH responsiveness. Give each batch a checkpoint/restart boundary and keep periodic status summaries compact.
- Start with one training job and bounded BLAS/OpenMP threads (for example 1–4 per process); avoid multiplying worker processes by unrestricted library thread pools. Keep the existing single-Isaac constraint.
- Budget local rollout/video/checkpoint growth. Retain the latest resumable checkpoint, best checkpoint per gate, associated proof, and referenced donor lineage. Git cleanup must not delete ignored training outputs, caches, or checkpoints.
- When SSH, GUI, or kernel health degrades, stop launching additional experiments and collect evidence before escalating load or repeatedly reconnecting. Preserve a running worker unless intervention is necessary for the authorized recovery.

## Preserve cloud I/O safeguards

On this TK2 installation, `/home/wishai/Nextcloud` is a read-only rclone mount owned by `pt-process-nextcloud-mount.service`. Its bounded local read-cache override is:

`~/.config/systemd/user/pt-process-nextcloud-mount.service.d/20-bounded-read-cache.conf`

Verify the live unit before changing it. The installed settings use full read caching, an 8 GiB soft cache limit, 24-hour retention, 8–32 MiB read chunks, a 1 MiB memory buffer, and a 30-second idle I/O timeout. Open files can temporarily exceed the soft limit. This cache consumes TK2 disk space; it does not duplicate data in Nextcloud.

Preserve `webgui/server.py` large-file streaming: headers precede cloud reads, body reads are bounded to 64 KiB, disconnects stop further reads, and the declared content length is respected. Keep `webgui/tests/test_large_file_streaming.py` passing when changing delivery or caching. Do not return to buffering whole large model/ZIP/video files before sending headers.

Reuse asset identities and cached data. Do not make a GUI refresh scan all cloud outputs or hash the same large file repeatedly. Cold or disconnected cloud reads need bounded retries/timeouts and must not accumulate abandoned HTTP request threads.

## Distinguish app, network, and host failures

- If only the app stalls, compare a small SSH command, local app-server request, and GUI request before resetting services.
- If SSH also fails, compare Tailscale presence, relay reachability, and the known LAN address/host key. A powered-on screen is not proof that the kernel or network is responsive.
- After a host reboot, inspect prior-boot kernel logs, including `journalctl -k -b -1`, for watchdog lockups, OOM, I/O, and Wi-Fi/driver errors. Capture the kernel version and relevant stack before assigning a cause.
- The 2026-11-06 outage included GUI threads blocked on uncached cloud reads earlier in the day. At 18:48 the previous boot logged an Intel `iwlwifi` hardware restart followed by hard and soft CPU lockups on kernel `6.8.0-147-generic`; TK2 then became unreachable on both LAN and Tailscale. A specific kernel/firmware defect has not been established. Prefer wired Ethernet for unattended work when available and treat recurring kernel/driver faults as host maintenance, not a Codex-only failure.
- Do not disable watchdogs, OOM protection, or other system safeguards to hide a recurrence. Do not claim that a short successful connectivity test proves half-day endurance.

## Record interruption truthfully

After a reboot, verify the worker exists before showing `running`. For the current ledger convention, use a diagnosis/interrupted state, preserve checkpoint hashes and prior proofs, and leave the earliest unresolved gate unresolved. Resume from validated artifacts through the normal launcher; a crash or restart does not certify a gate.

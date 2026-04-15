#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path("/remote-home/pxl/R2R")
    status_path = root / "output" / "entropy_online_trace_watcher" / "status.json"
    if len(sys.argv) > 1:
        status_path = Path(sys.argv[1])

    if not status_path.exists():
        print(f"status file not found: {status_path}")
        return 1

    with status_path.open("r") as handle:
        status = json.load(handle)

    print(f"Updated: {status.get('updated_at')}")
    print(f"Phase: {status.get('phase')}")
    print(f"Message: {status.get('message')}")
    print()

    watcher = status.get("watcher", {})
    experiment = status.get("experiment", {})
    child = status.get("child", {})

    print(f"Watcher PID: {watcher.get('pid')}")
    print(f"Requested GPUs: {watcher.get('gpu_count')}")
    print(f"Allowed GPUs: {watcher.get('allowed_gpu_indices') or '<all>'}")
    print(
        "Strict idle rule: "
        f"memory<={watcher.get('idle_memory_used_mib_max')} MiB, "
        f"util<={watcher.get('idle_utilization_gpu_max')}%, "
        f"confirm_rounds={watcher.get('gpu_idle_confirm_rounds')}"
    )
    print(f"Selected GPUs: {experiment.get('selected_gpu_indices') or '<none>'}")
    print(f"Output dir: {child.get('output_dir')}")
    print(f"Child PID: {child.get('pid')}")
    if child.get("exit_code") is not None:
        print(f"Child exit code: {child.get('exit_code')}")
    print()

    print("GPU Snapshot")
    for gpu in status.get("gpu_snapshot", []):
        reasons = ",".join(gpu.get("reasons_not_idle", [])) or "idle"
        print(
            f"  GPU {gpu.get('index')}: "
            f"mem={gpu.get('memory_used_mib')} MiB "
            f"util={gpu.get('utilization_gpu')}% "
            f"compute_pids={gpu.get('compute_pids')} "
            f"idle_rounds={gpu.get('idle_rounds')} "
            f"confirmed={gpu.get('idle_confirmed')} "
            f"reasons={reasons}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

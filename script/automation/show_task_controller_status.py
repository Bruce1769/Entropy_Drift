#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main():
    root = Path("/remote-home/pxl/R2R")
    status_path = root / "output" / "task_controller_state" / "controller_status.json"
    if len(sys.argv) > 1:
        status_path = Path(sys.argv[1])

    if not status_path.exists():
        print(f"status file not found: {status_path}")
        sys.exit(1)

    with status_path.open("r") as f:
        status = json.load(f)

    print(f"Updated: {status.get('updated_at')}")
    print()
    for task in status.get("tasks", []):
        state = task.get("state", {})
        schedule_mode = task.get("schedule_mode", "fixed")
        requested_gpu_count = task.get("requested_gpu_count")
        assigned_gpu_indices = task.get("assigned_gpu_indices", [])
        configured_gpu_indices = task.get("configured_gpu_indices", [])
        allowed_gpu_indices = task.get("allowed_gpu_indices", [])
        print(task["name"])
        print(f"  Schedule: {schedule_mode}")
        print(f"  Requested GPUs: {requested_gpu_count}")
        print(f"  Assigned GPUs: {assigned_gpu_indices or '<none>'}")
        if configured_gpu_indices:
            print(f"  Fixed GPUs: {configured_gpu_indices}")
        elif allowed_gpu_indices:
            print(f"  Allowed GPUs: {allowed_gpu_indices}")
        print(f"  PID: {state.get('pid')}")
        print(f"  Last action: {state.get('last_action')}")
        print(f"  Next phase: {task.get('next_phase')}")
        print(f"  Next problems: {','.join(task.get('next_problem_ids', [])) or '<none>'}")
        print(f"  Skip IDs: {','.join(task.get('skip_problem_ids', [])) or '<none>'}")
        print()


if __name__ == "__main__":
    main()

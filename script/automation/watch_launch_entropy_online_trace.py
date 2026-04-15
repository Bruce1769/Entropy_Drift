#!/usr/bin/env python3

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def is_live_pid(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False

    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return False
    try:
        for line in status_path.read_text().splitlines():
            if line.startswith("State:"):
                parts = line.split()
                return len(parts) >= 2 and parts[1] != "Z"
    except OSError:
        return False
    return False


def parse_gpu_indices(text: str) -> Optional[List[int]]:
    value = text.strip()
    if not value:
        return None
    indices: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            raise ValueError(f"invalid gpu index: {item}")
        indices.append(int(item))
    return indices or None


class EntropyOnlineTraceWatcher:
    def __init__(self, args: argparse.Namespace):
        self.repo_root = Path(args.repo_root).resolve()
        self.python_bin = str(Path(args.python_bin).resolve())
        self.script_path = Path(args.script_path).resolve()
        self.state_dir = Path(args.state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.status_path = self.state_dir / "status.json"
        self.log_path = self.state_dir / "watcher.log"
        self.experiment_log_path = self.state_dir / "experiment.log"

        self.dataset = args.dataset
        self.config_path = args.config_path
        self.entropy_threshold = float(args.entropy_threshold)
        self.output_dir = args.output_dir.strip()
        self.trace_reference_topk_k = int(args.trace_reference_topk_k)
        self.poll_interval_sec = int(args.poll_interval_sec)
        self.gpu_count = int(args.gpu_count)
        self.idle_memory_used_mib_max = float(args.idle_memory_used_mib_max)
        self.idle_utilization_gpu_max = float(args.idle_utilization_gpu_max)
        self.gpu_idle_confirm_rounds = int(args.gpu_idle_confirm_rounds)
        self.max_wait_sec = int(args.max_wait_sec) if args.max_wait_sec else 0
        self.allowed_gpu_indices = (
            sorted(set(args.allowed_gpu_indices))
            if args.allowed_gpu_indices
            else None
        )
        self.force_hf_online = bool(args.force_hf_online)

        self.watcher_started_at = time.time()
        self.child_proc: Optional[subprocess.Popen] = None
        self.child_pid: Optional[int] = None
        self.child_pgid: Optional[int] = None
        self.child_started_at: Optional[float] = None
        self.selected_gpu_indices: List[int] = []
        self.idle_rounds: Dict[int, int] = {}
        self.shutdown_requested = False
        self.final_phase: Optional[str] = None
        self.last_gpu_snapshot: List[Dict[str, object]] = []

        self._install_signal_handlers()
        self._guard_against_existing_child()

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, _frame) -> None:
        self.shutdown_requested = True
        self._log(f"Received signal {signum}, shutting down watcher")

    def _default_output_dir(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        threshold_tag = f"{self.entropy_threshold:.2f}".replace(".", "")
        return str(
            self.repo_root
            / "output"
            / "online_eval"
            / f"qwen3_0_6b_qwen3_8b_{self.dataset}_entropy_t{threshold_tag}_online_trace_{timestamp}"
        )

    def _guard_against_existing_child(self) -> None:
        if not self.status_path.exists():
            return
        try:
            status = json.loads(self.status_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        child_pid = status.get("child", {}).get("pid")
        if child_pid and is_live_pid(int(child_pid)):
            raise RuntimeError(
                f"existing launched experiment pid {child_pid} is still running; "
                f"state_dir={self.state_dir}"
            )

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.log_path.open("a") as handle:
            handle.write(line + "\n")

    def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

    def _gpu_uuid_map(self) -> Dict[int, str]:
        result = self._run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ]
        )
        mapping: Dict[int, str] = {}
        if result.returncode != 0:
            return mapping

        for raw_line in result.stdout.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            mapping[int(parts[0])] = parts[1]
        return mapping

    def _gpu_compute_map(self) -> Dict[int, List[int]]:
        uuid_map = self._gpu_uuid_map()
        reverse_uuid_map = {uuid: index for index, uuid in uuid_map.items()}
        compute_map: Dict[int, List[int]] = {index: [] for index in uuid_map}

        result = self._run_command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        )
        if result.returncode != 0:
            return compute_map

        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line or "No running processes found" in line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            gpu_index = reverse_uuid_map.get(parts[0])
            if gpu_index is None:
                continue
            compute_map.setdefault(gpu_index, []).append(int(parts[1]))
        return compute_map

    def _gpu_stats_map(self) -> Dict[int, Dict[str, float]]:
        result = self._run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        stats_map: Dict[int, Dict[str, float]] = {}
        if result.returncode != 0:
            return stats_map

        for raw_line in result.stdout.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            try:
                index = int(parts[0])
                memory_used_mib = float(parts[1])
                memory_total_mib = float(parts[2])
                utilization_gpu = float(parts[3])
            except ValueError:
                continue
            stats_map[index] = {
                "memory_used_mib": memory_used_mib,
                "memory_total_mib": memory_total_mib,
                "utilization_gpu": utilization_gpu,
            }
        return stats_map

    def _build_gpu_snapshot(self) -> List[Dict[str, object]]:
        compute_map = self._gpu_compute_map()
        stats_map = self._gpu_stats_map()
        all_gpu_indices = sorted(set(compute_map.keys()) | set(stats_map.keys()))
        snapshot: List[Dict[str, object]] = []

        allowed_set = set(self.allowed_gpu_indices) if self.allowed_gpu_indices is not None else None
        for index in all_gpu_indices:
            stats = stats_map.get(index, {})
            compute_pids = compute_map.get(index, [])
            memory_used_mib = float(stats.get("memory_used_mib", 10**9))
            memory_total_mib = float(stats.get("memory_total_mib", 0))
            utilization_gpu = float(stats.get("utilization_gpu", 10**9))
            reasons: List[str] = []

            if allowed_set is not None and index not in allowed_set:
                reasons.append("not_allowed")
            if compute_pids:
                reasons.append("compute_busy")
            if memory_used_mib > self.idle_memory_used_mib_max:
                reasons.append("memory_busy")
            if utilization_gpu > self.idle_utilization_gpu_max:
                reasons.append("util_busy")

            truly_idle_now = not reasons
            if truly_idle_now:
                self.idle_rounds[index] = self.idle_rounds.get(index, 0) + 1
            else:
                self.idle_rounds[index] = 0

            snapshot.append(
                {
                    "index": index,
                    "memory_used_mib": memory_used_mib,
                    "memory_total_mib": memory_total_mib,
                    "utilization_gpu": utilization_gpu,
                    "compute_pids": compute_pids,
                    "truly_idle_now": truly_idle_now,
                    "idle_rounds": self.idle_rounds.get(index, 0),
                    "idle_confirmed": self.idle_rounds.get(index, 0)
                    >= self.gpu_idle_confirm_rounds,
                    "reasons_not_idle": reasons,
                }
            )

        for index in list(self.idle_rounds):
            if index not in all_gpu_indices:
                self.idle_rounds.pop(index, None)

        return snapshot

    def _select_gpu_indices(self, snapshot: List[Dict[str, object]]) -> Optional[List[int]]:
        confirmed = [
            gpu
            for gpu in snapshot
            if bool(gpu["idle_confirmed"])
        ]
        if len(confirmed) < self.gpu_count:
            return None

        confirmed.sort(
            key=lambda gpu: (
                float(gpu["memory_used_mib"]),
                float(gpu["utilization_gpu"]),
                int(gpu["index"]),
            )
        )
        return [int(gpu["index"]) for gpu in confirmed[: self.gpu_count]]

    def _build_child_env(self, gpu_indices: List[int]) -> Dict[str, str]:
        env = os.environ.copy()
        env["PYTHON_BIN"] = self.python_bin
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
        env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
        env["DATASET"] = self.dataset
        env["CONFIG_PATH"] = self.config_path
        env["OUTPUT_DIR"] = self.output_dir
        env["ENTROPY_THRESHOLD"] = str(self.entropy_threshold)
        env["TRACE_REFERENCE_TOPK_K"] = str(self.trace_reference_topk_k)
        if self.force_hf_online:
            env["FORCE_HF_ONLINE"] = "1"
        return env

    def _write_status(
        self,
        phase: str,
        message: str,
        exit_code: Optional[int] = None,
    ) -> None:
        child_status = {
            "pid": self.child_pid,
            "pgid": self.child_pgid,
            "started_at": self.child_started_at,
            "output_dir": self.output_dir,
            "script_path": str(self.script_path),
            "log_path": str(self.experiment_log_path),
        }
        if exit_code is not None:
            child_status["exit_code"] = exit_code

        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at_epoch": time.time(),
            "phase": phase,
            "message": message,
            "watcher": {
                "started_at": self.watcher_started_at,
                "pid": os.getpid(),
                "repo_root": str(self.repo_root),
                "state_dir": str(self.state_dir),
                "poll_interval_sec": self.poll_interval_sec,
                "gpu_count": self.gpu_count,
                "allowed_gpu_indices": self.allowed_gpu_indices,
                "gpu_idle_confirm_rounds": self.gpu_idle_confirm_rounds,
                "idle_memory_used_mib_max": self.idle_memory_used_mib_max,
                "idle_utilization_gpu_max": self.idle_utilization_gpu_max,
                "max_wait_sec": self.max_wait_sec,
            },
            "experiment": {
                "dataset": self.dataset,
                "config_path": self.config_path,
                "entropy_threshold": self.entropy_threshold,
                "trace_reference_topk_k": self.trace_reference_topk_k,
                "selected_gpu_indices": self.selected_gpu_indices,
            },
            "child": child_status,
            "gpu_snapshot": self.last_gpu_snapshot,
        }
        with self.status_path.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _launch_child(self, gpu_indices: List[int]) -> None:
        if not self.output_dir:
            self.output_dir = self._default_output_dir()
        env = self._build_child_env(gpu_indices)
        self.selected_gpu_indices = list(gpu_indices)
        with self.experiment_log_path.open("a") as handle:
            handle.write(
                f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"launch CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
                f"OUTPUT_DIR={self.output_dir} ====\n"
            )

        log_handle = self.experiment_log_path.open("a")
        try:
            proc = subprocess.Popen(
                ["/bin/bash", str(self.script_path)],
                cwd=str(self.repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()

        self.child_proc = proc
        self.child_pid = proc.pid
        self.child_started_at = time.time()
        try:
            self.child_pgid = os.getpgid(proc.pid)
        except OSError:
            self.child_pgid = None

        self._log(
            "Launched entropy online trace "
            f"pid={self.child_pid} gpus={','.join(str(index) for index in gpu_indices)} "
            f"output_dir={self.output_dir}"
        )

    def _terminate_child(self) -> None:
        if not self.child_pid or not is_live_pid(self.child_pid):
            return

        self._log(f"Stopping launched experiment pid={self.child_pid}")
        if self.child_pgid is not None:
            try:
                os.killpg(self.child_pgid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.time() + 30
            while time.time() < deadline:
                if not is_live_pid(self.child_pid):
                    return
                time.sleep(1)
            try:
                os.killpg(self.child_pgid, signal.SIGKILL)
            except OSError:
                pass
            return

        try:
            os.kill(self.child_pid, signal.SIGTERM)
        except OSError:
            pass

    def run(self) -> int:
        self._log(
            "Watching for truly idle GPUs "
            f"(count={self.gpu_count}, allowed={self.allowed_gpu_indices or 'all'}, "
            f"memory<={self.idle_memory_used_mib_max} MiB, "
            f"util<={self.idle_utilization_gpu_max}%, "
            f"confirm_rounds={self.gpu_idle_confirm_rounds})"
        )

        while True:
            if self.shutdown_requested:
                if self.child_pid and is_live_pid(self.child_pid):
                    self._terminate_child()
                self.final_phase = "stopped"
                self._write_status("stopped", "watcher stopped by signal")
                return 1

            if self.child_proc is None:
                self.last_gpu_snapshot = self._build_gpu_snapshot()
                selected = self._select_gpu_indices(self.last_gpu_snapshot)
                if selected is not None:
                    self._write_status(
                        "launching",
                        f"found confirmed idle gpus {selected}, launching experiment",
                    )
                    self._launch_child(selected)
                    self._write_status(
                        "running",
                        f"experiment launched on gpus {selected}",
                    )
                else:
                    waited_sec = int(time.time() - self.watcher_started_at)
                    self._write_status(
                        "waiting_for_idle_gpus",
                        f"waiting for {self.gpu_count} truly idle gpus for {waited_sec}s",
                    )
                    if self.max_wait_sec and waited_sec >= self.max_wait_sec:
                        self.final_phase = "timed_out"
                        self._log("Timed out while waiting for idle GPUs")
                        self._write_status("timed_out", "timed out waiting for idle gpus")
                        return 1
            else:
                exit_code = self.child_proc.poll()
                if exit_code is None:
                    self._write_status(
                        "running",
                        f"experiment pid {self.child_pid} is running",
                    )
                else:
                    phase = "completed" if exit_code == 0 else "failed"
                    self.final_phase = phase
                    self._log(f"Experiment exited with code {exit_code}")
                    self._write_status(
                        phase,
                        f"experiment exited with code {exit_code}",
                        exit_code=exit_code,
                    )
                    return 0 if exit_code == 0 else exit_code

            time.sleep(self.poll_interval_sec)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch for two truly idle GPUs, then launch entropy online trace."
    )
    parser.add_argument(
        "--repo-root",
        default="/remote-home/pxl/R2R",
        help="Repository root.",
    )
    parser.add_argument(
        "--python-bin",
        default="/remote-home/pxl/miniconda3/envs/r2r/bin/python",
        help="Python interpreter passed into the experiment shell script.",
    )
    parser.add_argument(
        "--script-path",
        default="/remote-home/pxl/R2R/script/evaluate/run_qwen3_0_6b_qwen3_8b_aime26_entropy_online_trace.sh",
        help="Shell script to launch once enough GPUs are idle.",
    )
    parser.add_argument(
        "--state-dir",
        default="/remote-home/pxl/R2R/output/entropy_online_trace_watcher",
        help="Directory for watcher logs and state.",
    )
    parser.add_argument(
        "--dataset",
        default="aime26",
        help="Dataset name passed to the experiment script.",
    )
    parser.add_argument(
        "--config-path",
        default="config/Qwen3-0.6B+Qwen3-8B_entropy.yaml",
        help="Model config path passed to the experiment script.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional fixed output dir. Defaults to a timestamped online_eval path.",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.45,
        help="Entropy threshold used by the experiment script.",
    )
    parser.add_argument(
        "--trace-reference-topk-k",
        type=int,
        default=64,
        help="trace_reference_topk_k passed to the experiment script.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=2,
        help="How many GPUs must be truly idle before launch.",
    )
    parser.add_argument(
        "--allowed-gpu-indices",
        type=parse_gpu_indices,
        default=None,
        help="Optional comma-separated whitelist, for example 4,5,6,7.",
    )
    parser.add_argument(
        "--idle-memory-used-mib-max",
        type=float,
        default=1024.0,
        help="Maximum residual memory usage for a GPU to count as truly idle.",
    )
    parser.add_argument(
        "--idle-utilization-gpu-max",
        type=float,
        default=5.0,
        help="Maximum GPU utilization percentage for a GPU to count as truly idle.",
    )
    parser.add_argument(
        "--gpu-idle-confirm-rounds",
        type=int,
        default=2,
        help="How many consecutive polls a GPU must satisfy the strict idle rule.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=60,
        help="Polling interval for nvidia-smi.",
    )
    parser.add_argument(
        "--max-wait-sec",
        type=int,
        default=0,
        help="Optional timeout. Zero means wait indefinitely.",
    )
    parser.add_argument(
        "--force-hf-online",
        action="store_true",
        help="Unset HF offline mode inside the launched experiment script.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        watcher = EntropyOnlineTraceWatcher(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return watcher.run()


if __name__ == "__main__":
    sys.exit(main())

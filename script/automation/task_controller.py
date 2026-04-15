#!/usr/bin/env python3

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class PhaseInfo:
    name: str
    pending_problem_ids: List[str]
    max_new_tokens: int


class TaskController:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with self.config_path.open("r") as f:
            self.config = json.load(f)

        self.repo_root = Path(self.config["repo_root"])
        self.python_bin = self.config["python_bin"]
        self.state_dir = Path(self.config["state_dir"])
        self.logs_dir = self.state_dir / "logs"
        self.tasks_state_dir = self.state_dir / "tasks"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_state_dir.mkdir(parents=True, exist_ok=True)

        self.poll_interval_sec = int(self.config.get("poll_interval_sec", 60))
        self.gpu_idle_confirm_rounds = int(self.config.get("gpu_idle_confirm_rounds", 2))
        self.startup_grace_sec = int(self.config.get("startup_grace_sec", 900))
        self.stall_timeout_sec = int(self.config.get("stall_timeout_sec", 900))
        self.failure_cooldown_sec = int(self.config.get("failure_cooldown_sec", 90))
        self.min_runtime_before_failure_sec = int(
            self.config.get("min_runtime_before_failure_sec", 300)
        )
        self.max_consecutive_failures_before_skip = int(
            self.config.get("max_consecutive_failures_before_skip", 2)
        )

        self.default_env = dict(self.config.get("default_env", {}))
        self.tasks = [task for task in self.config.get("tasks", []) if task.get("enabled", True)]
        self.procs: Dict[str, subprocess.Popen] = {}
        self.gpu_idle_rounds: Dict[int, int] = {}

        self._initialize_task_state_files()

    def _pid_state(self, pid: Optional[int]) -> Optional[str]:
        if not pid:
            return None

        status_path = Path(f"/proc/{pid}/status")
        if not status_path.exists():
            return None

        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("State:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
                    return None
        except OSError:
            return None

        return None

    def _initialize_task_state_files(self) -> None:
        for task in self.tasks:
            task_state = self._load_task_state(task)
            if not task_state:
                task_state = {
                    "pid": None,
                    "phase": None,
                    "started_at": None,
                    "cooldown_until": 0,
                    "consecutive_failures": 0,
                    "last_action": "initialized",
                }
                self._save_task_state(task, task_state)

            skip_path = self._skip_file_path(task)
            if not skip_path.exists():
                initial_skip_ids = [str(x) for x in task.get("initial_skip_problem_ids", [])]
                if initial_skip_ids:
                    skip_path.write_text("\n".join(initial_skip_ids) + "\n")
                else:
                    skip_path.write_text("")

    def _task_dir(self, task: Dict) -> Path:
        return self.tasks_state_dir / task["name"]

    def _task_log_path(self, task: Dict) -> Path:
        return self.logs_dir / f"{task['name']}.log"

    def _task_state_path(self, task: Dict) -> Path:
        return self._task_dir(task) / "state.json"

    def _skip_file_path(self, task: Dict) -> Path:
        task_dir = self._task_dir(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir / "skip_problem_ids.txt"

    def _load_task_state(self, task: Dict) -> Dict:
        state_path = self._task_state_path(task)
        if not state_path.exists():
            return {}
        with state_path.open("r") as f:
            return json.load(f)

    def _save_task_state(self, task: Dict, state: Dict) -> None:
        state_path = self._task_state_path(task)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("w") as f:
            json.dump(state, f, indent=2, sort_keys=True)

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with (self.logs_dir / "controller.log").open("a") as f:
            f.write(line + "\n")

    def _task_log(self, task: Dict, message: str) -> None:
        self._log(f"{task['name']}: {message}")

    def _parse_problem_ids(self, task: Dict) -> List[str]:
        if "problem_ids" in task:
            return [str(x) for x in task["problem_ids"]]
        if "problem_id_range" in task:
            start, end = task["problem_id_range"]
            return [str(i) for i in range(int(start), int(end) + 1)]
        raise ValueError(f"Task {task['name']} missing problem_ids/problem_id_range")

    def _problem_order(self, task: Dict) -> Dict[str, int]:
        return {problem_id: idx for idx, problem_id in enumerate(self._parse_problem_ids(task))}

    def _read_skip_problem_ids(self, task: Dict) -> List[str]:
        skip_path = self._skip_file_path(task)
        if not skip_path.exists():
            return []
        return [line.strip() for line in skip_path.read_text().splitlines() if line.strip()]

    def _append_skip_problem_id(self, task: Dict, problem_id: str) -> None:
        skip_ids = self._read_skip_problem_ids(task)
        if problem_id in skip_ids:
            return
        with self._skip_file_path(task).open("a") as f:
            f.write(str(problem_id) + "\n")
        self._task_log(task, f"Added problem {problem_id} to skip list")

    def _get_completed_problem_ids(self, output_dir: Path) -> List[str]:
        temp_dir = output_dir / "temp"
        if not temp_dir.exists():
            return []
        completed = set()
        for path in temp_dir.glob("*_run_*.txt"):
            completed.add(path.name.split("_run_")[0])
        return sorted(completed, key=self._natural_key)

    def _get_missing_answer_problem_ids(self, output_dir: Path) -> List[str]:
        helper_path = self.repo_root / "script" / "evaluate" / "find_missing_answer_problem_ids.py"
        if not helper_path.exists():
            return []
        result = subprocess.run(
            [self.python_bin, str(helper_path), "--output_dir", str(output_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        text = result.stdout.strip()
        if not text:
            return []
        return [item.strip() for item in text.split(",") if item.strip()]

    def _natural_key(self, text: str):
        return (0, int(text)) if str(text).isdigit() else (1, str(text))

    def _sort_problem_ids(self, task: Dict, problem_ids: List[str]) -> List[str]:
        order = self._problem_order(task)
        return sorted(problem_ids, key=lambda pid: (order.get(pid, 10**9), self._natural_key(pid)))

    def _get_phase_info(self, task: Dict) -> Optional[PhaseInfo]:
        output_dir = Path(task["output_dir"])
        all_problem_ids = self._parse_problem_ids(task)
        completed = set(self._get_completed_problem_ids(output_dir))
        skipped = set(self._read_skip_problem_ids(task))

        main_pending = [pid for pid in all_problem_ids if pid not in completed and pid not in skipped]
        if main_pending:
            return PhaseInfo(
                name="main",
                pending_problem_ids=main_pending,
                max_new_tokens=int(task["main_max_new_tokens"]),
            )

        missing_answer_ids = self._get_missing_answer_problem_ids(output_dir)
        rerun_pending = [pid for pid in self._sort_problem_ids(task, missing_answer_ids) if pid not in skipped]
        if rerun_pending:
            return PhaseInfo(
                name="rerun",
                pending_problem_ids=rerun_pending,
                max_new_tokens=int(task["rerun_max_new_tokens"]),
            )

        return None

    def _latest_progress_mtime(self, task: Dict) -> float:
        output_dir = Path(task["output_dir"])
        candidates = [self._task_log_path(task)]

        for subdir_name in ("temp", "temp_csv"):
            subdir = output_dir / subdir_name
            if subdir.exists():
                candidates.extend(path for path in subdir.iterdir() if path.is_file())

        mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
        return max(mtimes) if mtimes else 0.0

    def _task_schedule_mode(self, task: Dict) -> str:
        mode = str(task.get("schedule_mode", "")).strip().lower()
        if mode:
            return mode
        return "fixed" if "gpu_indices" in task else "dynamic"

    def _task_requested_gpu_count(self, task: Dict) -> int:
        if self._task_schedule_mode(task) == "fixed":
            return len(self._task_fixed_gpu_indices(task))
        return int(task.get("gpu_count", 2))

    def _task_fixed_gpu_indices(self, task: Dict) -> List[int]:
        return [int(index) for index in task.get("gpu_indices", [])]

    def _all_gpu_indices(self, gpu_process_map: Dict[int, List[str]]) -> List[int]:
        return sorted(int(index) for index in gpu_process_map.keys())

    def _task_allowed_gpu_indices(
        self, task: Dict, all_gpu_indices: List[int]
    ) -> List[int]:
        if self._task_schedule_mode(task) == "fixed":
            return self._task_fixed_gpu_indices(task)
        if "allowed_gpu_indices" in task:
            return [int(index) for index in task["allowed_gpu_indices"]]
        return list(all_gpu_indices)

    def _read_process_env_var(self, pid: Optional[int], key: str) -> Optional[str]:
        if not pid:
            return None

        environ_path = Path(f"/proc/{pid}/environ")
        if not environ_path.exists():
            return None

        try:
            raw = environ_path.read_bytes()
        except OSError:
            return None

        prefix = f"{key}=".encode()
        for item in raw.split(b"\0"):
            if item.startswith(prefix):
                try:
                    return item[len(prefix) :].decode()
                except UnicodeDecodeError:
                    return None
        return None

    def _assigned_gpu_indices(self, task: Dict, state: Dict) -> List[int]:
        assigned = state.get("assigned_gpu_indices") or []
        if assigned:
            return [int(index) for index in assigned]

        fixed_gpu_indices = self._task_fixed_gpu_indices(task)
        if fixed_gpu_indices:
            return fixed_gpu_indices

        visible = self._read_process_env_var(state.get("pid"), "CUDA_VISIBLE_DEVICES")
        if not visible:
            return []

        gpu_indices: List[int] = []
        for item in visible.split(","):
            item = item.strip()
            if item.isdigit():
                gpu_indices.append(int(item))
        return gpu_indices

    def _gpu_uuid_map(self) -> Dict[int, str]:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        mapping: Dict[int, str] = {}
        if result.returncode != 0:
            return mapping
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                mapping[int(parts[0])] = parts[1]
        return mapping

    def _gpu_process_map(self) -> Dict[int, List[str]]:
        uuid_map = self._gpu_uuid_map()
        reverse_uuid_map = {uuid: index for index, uuid in uuid_map.items()}
        gpu_processes: Dict[int, List[str]] = {index: [] for index in uuid_map}

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return gpu_processes

        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            gpu_uuid, pid = parts[0], parts[1]
            gpu_index = reverse_uuid_map.get(gpu_uuid)
            if gpu_index is not None:
                gpu_processes.setdefault(gpu_index, []).append(pid)
        return gpu_processes

    def _gpu_stats_map(self) -> Dict[int, Dict[str, float]]:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        stats: Dict[int, Dict[str, float]] = {}
        if result.returncode != 0:
            return stats

        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            index = int(parts[0])
            try:
                memory_used_mib = float(parts[1])
                memory_total_mib = float(parts[2])
                utilization_gpu = float(parts[3])
            except ValueError:
                continue
            stats[index] = {
                "memory_used_mib": memory_used_mib,
                "memory_total_mib": memory_total_mib,
                "utilization_gpu": utilization_gpu,
            }
        return stats

    def _gpu_free_gib(self, gpu_stats_map: Dict[int, Dict[str, float]], gpu_index: int) -> float:
        stats = gpu_stats_map.get(gpu_index)
        if not stats:
            return 0.0
        return max(stats["memory_total_mib"] - stats["memory_used_mib"], 0.0) / 1024.0

    def _fixed_partial_quick_gpu_ready(
        self,
        task: Dict,
        gpu_indices: List[int],
        gpu_process_map: Dict[int, List[str]],
        gpu_stats_map: Dict[int, Dict[str, float]],
        reserved_gpu_indices: List[int],
    ) -> bool:
        if not task.get("allow_partial_quick_gpu") or len(gpu_indices) < 2:
            return False

        reserved = set(int(index) for index in reserved_gpu_indices)
        if any(index in reserved for index in gpu_indices):
            return False

        quick_gpu_index = int(gpu_indices[0])
        reference_gpu_indices = [int(index) for index in gpu_indices[1:]]
        quick_max_utilization = float(task.get("partial_quick_gpu_max_utilization", 5))
        quick_min_free_gib = float(task.get("partial_quick_gpu_min_free_gib", 12))
        reference_min_free_gib = float(task.get("reference_gpu_min_free_gib", 20))
        reference_max_utilization = float(task.get("reference_gpu_max_utilization", 5))

        quick_stats = gpu_stats_map.get(quick_gpu_index)
        if not quick_stats:
            return False
        if quick_stats["utilization_gpu"] > quick_max_utilization:
            return False
        if self._gpu_free_gib(gpu_stats_map, quick_gpu_index) < quick_min_free_gib:
            return False

        for reference_gpu_index in reference_gpu_indices:
            reference_stats = gpu_stats_map.get(reference_gpu_index)
            if not reference_stats:
                return False
            if gpu_process_map.get(reference_gpu_index, []):
                return False
            if reference_stats["utilization_gpu"] > reference_max_utilization:
                return False
            if self._gpu_free_gib(gpu_stats_map, reference_gpu_index) < reference_min_free_gib:
                return False
            if self.gpu_idle_rounds.get(reference_gpu_index, 0) < self.gpu_idle_confirm_rounds:
                return False

        return True

    def _dynamic_partial_quick_gpu_selection(
        self,
        task: Dict,
        allowed_gpu_indices: List[int],
        gpu_process_map: Dict[int, List[str]],
        gpu_stats_map: Dict[int, Dict[str, float]],
        reserved_gpu_indices: List[int],
    ) -> Optional[List[int]]:
        if not task.get("allow_partial_quick_gpu"):
            return None

        gpu_count = self._task_requested_gpu_count(task)
        if gpu_count < 2:
            return None

        reserved = set(int(index) for index in reserved_gpu_indices)
        quick_max_utilization = float(task.get("partial_quick_gpu_max_utilization", 5))
        quick_min_free_gib = float(task.get("partial_quick_gpu_min_free_gib", 12))
        reference_min_free_gib = float(task.get("reference_gpu_min_free_gib", 20))
        reference_max_utilization = float(task.get("reference_gpu_max_utilization", 5))

        for quick_gpu_index in allowed_gpu_indices:
            if quick_gpu_index in reserved:
                continue

            quick_stats = gpu_stats_map.get(quick_gpu_index)
            if not quick_stats:
                continue
            if quick_stats["utilization_gpu"] > quick_max_utilization:
                continue
            if self._gpu_free_gib(gpu_stats_map, quick_gpu_index) < quick_min_free_gib:
                continue

            reference_gpu_indices: List[int] = []
            for reference_gpu_index in allowed_gpu_indices:
                if reference_gpu_index == quick_gpu_index or reference_gpu_index in reserved:
                    continue

                reference_stats = gpu_stats_map.get(reference_gpu_index)
                if not reference_stats:
                    continue
                if gpu_process_map.get(reference_gpu_index, []):
                    continue
                if reference_stats["utilization_gpu"] > reference_max_utilization:
                    continue
                if self._gpu_free_gib(gpu_stats_map, reference_gpu_index) < reference_min_free_gib:
                    continue
                if self.gpu_idle_rounds.get(reference_gpu_index, 0) < self.gpu_idle_confirm_rounds:
                    continue

                reference_gpu_indices.append(reference_gpu_index)
                if len(reference_gpu_indices) == gpu_count - 1:
                    return [quick_gpu_index] + reference_gpu_indices

        return None

    def _gpu_set_busy(
        self, gpu_indices: List[int], gpu_process_map: Dict[int, List[str]]
    ) -> bool:
        return any(gpu_process_map.get(int(index), []) for index in gpu_indices)

    def _process_alive(self, pid: Optional[int]) -> bool:
        if not pid:
            return False

        pid_state = self._pid_state(pid)
        if pid_state == "Z":
            return False

        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _read_process_cmdline(self, pid: Optional[int]) -> List[str]:
        if not pid:
            return []

        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if not cmdline_path.exists():
            return []

        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            return []

        return [part.decode(errors="ignore") for part in raw.split(b"\0") if part]

    def _task_matches_pid(self, task: Dict, pid: Optional[int]) -> bool:
        if not self._process_alive(pid):
            return False

        cmdline = self._read_process_cmdline(pid)
        if not cmdline:
            return False

        joined = " ".join(cmdline)
        if "script/evaluate/hf_dataset_sglang.py" not in joined:
            return False

        output_dir = str(task["output_dir"])
        for index, arg in enumerate(cmdline):
            if arg == "--output_dir" and index + 1 < len(cmdline) and cmdline[index + 1] == output_dir:
                return True

        return False

    def _matching_task_pids(self, task: Dict) -> List[int]:
        matches: List[int] = []
        for proc_dir in Path("/proc").iterdir():
            if not proc_dir.name.isdigit():
                continue
            pid = int(proc_dir.name)
            if self._task_matches_pid(task, pid):
                matches.append(pid)
        return sorted(matches)

    def _task_progress_is_recent(self, task: Dict) -> bool:
        latest_progress = self._latest_progress_mtime(task)
        if latest_progress <= 0:
            return False
        recent_window = max(self.startup_grace_sec, self.stall_timeout_sec)
        return (time.time() - latest_progress) <= recent_window

    def _adopt_running_task(self, task: Dict, pid: int) -> None:
        state = self._load_task_state(task)
        phase_info = self._get_phase_info(task)
        state.update(
            {
                "pid": pid,
                "phase": phase_info.name if phase_info else state.get("phase"),
                "started_at": state.get("started_at") or time.time(),
                "cooldown_until": 0,
                "assigned_gpu_indices": self._assigned_gpu_indices(task, {"pid": pid}),
                "last_action": f"adopted_{pid}",
                "log_path": str(self._task_log_path(task)),
            }
        )
        self._save_task_state(task, state)

    def _reconcile_task_processes(self, task: Dict) -> None:
        state = self._load_task_state(task)
        current_pid = state.get("pid")
        matching_pids = self._matching_task_pids(task)

        if current_pid and current_pid in matching_pids:
            for pid in matching_pids:
                if pid == current_pid:
                    continue
                try:
                    os.kill(pid, signal.SIGTERM)
                    self._task_log(task, f"Stopped duplicate pid {pid}; keeping pid {current_pid}")
                except OSError:
                    pass
            return

        if current_pid and self._task_matches_pid(task, current_pid):
            return

        if current_pid and not self._task_matches_pid(task, current_pid):
            state.update(
                {
                    "pid": None,
                    "phase": None,
                    "started_at": None,
                    "assigned_gpu_indices": [],
                    "last_action": "cleared_stale_pid",
                }
            )
            self._save_task_state(task, state)

        if not matching_pids:
            return

        if not self._task_progress_is_recent(task):
            self._task_log(task, "Ignoring matching pid(s) because task progress is stale")
            return

        keep_pid = matching_pids[-1]
        for pid in matching_pids[:-1]:
            try:
                os.kill(pid, signal.SIGTERM)
                self._task_log(task, f"Stopped duplicate pid {pid}; adopting pid {keep_pid}")
            except OSError:
                pass

        self._adopt_running_task(task, keep_pid)
        self._task_log(task, f"Adopted existing pid {keep_pid}")

    def _kill_task_process(self, task: Dict, reason: str) -> None:
        state = self._load_task_state(task)
        pid = state.get("pid")
        if not pid:
            return
        self._task_log(task, f"Stopping pid {pid} due to {reason}")
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pgid = None

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.time() + 60
            while time.time() < deadline:
                if not self._process_alive(pid):
                    break
                time.sleep(1)
            if self._process_alive(pid):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

        self.procs.pop(task["name"], None)

    def _build_env(self, task: Dict, phase_info: PhaseInfo, gpu_indices: List[int]) -> Dict[str, str]:
        env = os.environ.copy()
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
            if key not in self.default_env and key not in task.get("extra_env", {}):
                env.pop(key, None)
        env.update(self.default_env)
        env.update(task.get("extra_env", {}))
        env["PYTHON_BIN"] = self.python_bin
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
        env["OUTPUT_DIR"] = task["output_dir"]
        env["MAX_NEW_TOKENS"] = str(phase_info.max_new_tokens)
        env["SKIP_PROBLEM_IDS"] = ",".join(self._read_skip_problem_ids(task))
        env["RERUN_MISSING_ANSWERS"] = "0"

        if phase_info.name == "main":
            env["RESUME"] = "1"
            env.pop("PROBLEM_IDS", None)
        else:
            env["RESUME"] = "0"
            env["PROBLEM_IDS"] = ",".join(phase_info.pending_problem_ids)

        return env

    def _start_task(self, task: Dict, phase_info: PhaseInfo, gpu_indices: List[int]) -> None:
        log_path = self._task_log_path(task)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = self._build_env(task, phase_info, gpu_indices)
        with log_path.open("a") as f:
            f.write(
                f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} start {phase_info.name} "
                f"on GPUs {env['CUDA_VISIBLE_DEVICES']} max_new_tokens={phase_info.max_new_tokens} ====\n"
            )

        log_handle = log_path.open("a")
        try:
            proc = subprocess.Popen(
                ["/bin/bash", task["script_path"]],
                cwd=str(self.repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        self.procs[task["name"]] = proc
        state = self._load_task_state(task)
        state.update(
            {
                "pid": proc.pid,
                "phase": phase_info.name,
                "started_at": time.time(),
                "cooldown_until": 0,
                "consecutive_failures": state.get("consecutive_failures", 0),
                "last_action": f"started_{phase_info.name}",
                "last_command_problem_ids": phase_info.pending_problem_ids,
                "assigned_gpu_indices": gpu_indices,
                "log_path": str(log_path),
            }
        )
        self._save_task_state(task, state)
        self._task_log(
            task,
            f"Started phase={phase_info.name} pid={proc.pid} gpus={','.join(str(i) for i in gpu_indices)} "
            f"pending={','.join(phase_info.pending_problem_ids)}",
        )

    def _handle_stall(self, task: Dict, state: Dict, phase_info: PhaseInfo) -> None:
        current_problem_id = phase_info.pending_problem_ids[0] if phase_info.pending_problem_ids else None
        if current_problem_id:
            self._append_skip_problem_id(task, current_problem_id)
        self._kill_task_process(task, reason="stall")
        state.update(
            {
                "pid": None,
                "phase": None,
                "started_at": None,
                "cooldown_until": time.time() + self.failure_cooldown_sec,
                "consecutive_failures": 0,
                "assigned_gpu_indices": [],
                "last_action": f"stalled_skip_{current_problem_id or 'unknown'}",
            }
        )
        self._save_task_state(task, state)

    def _handle_process_exit(self, task: Dict, state: Dict, exit_code: int) -> None:
        phase_info = self._get_phase_info(task)
        runtime = 0
        if state.get("started_at"):
            runtime = time.time() - float(state["started_at"])

        if phase_info is None:
            state.update(
                {
                    "pid": None,
                    "phase": None,
                    "started_at": None,
                    "cooldown_until": 0,
                    "consecutive_failures": 0,
                    "assigned_gpu_indices": [],
                    "last_action": f"completed_exit_{exit_code}",
                }
            )
            self._save_task_state(task, state)
            self._task_log(task, f"Finished all work with exit_code={exit_code}")
            return

        failures = int(state.get("consecutive_failures", 0))
        if exit_code != 0 and runtime < self.min_runtime_before_failure_sec:
            failures += 1
            current_problem_id = phase_info.pending_problem_ids[0] if phase_info.pending_problem_ids else None
            if failures >= self.max_consecutive_failures_before_skip and current_problem_id:
                self._append_skip_problem_id(task, current_problem_id)
                failures = 0
                last_action = f"exit_skip_{current_problem_id}"
                self._task_log(
                    task,
                    f"Process exited repeatedly on problem {current_problem_id}; added to skip list",
                )
            else:
                last_action = f"exit_retry_{exit_code}"
        else:
            failures = 0
            last_action = f"exit_retry_{exit_code}"

        state.update(
            {
                "pid": None,
                "phase": None,
                "started_at": None,
                "cooldown_until": time.time() + self.failure_cooldown_sec,
                "consecutive_failures": failures,
                "assigned_gpu_indices": [],
                "last_action": last_action,
            }
        )
        self._save_task_state(task, state)
        self._task_log(task, f"Process exited with exit_code={exit_code}; pending phase remains")

    def _monitor_running_task(self, task: Dict) -> None:
        state = self._load_task_state(task)
        pid = state.get("pid")
        if not pid:
            return

        proc = self.procs.get(task["name"])
        if proc is not None:
            exit_code = proc.poll()
            if exit_code is not None:
                self.procs.pop(task["name"], None)
                self._handle_process_exit(task, state, exit_code)
                return

        if not self._process_alive(pid):
            self.procs.pop(task["name"], None)
            self._handle_process_exit(task, state, exit_code=1)
            return

        started_at = float(state.get("started_at") or time.time())
        last_progress = self._latest_progress_mtime(task)
        elapsed_since_progress = time.time() - max(last_progress, started_at)
        grace = self.startup_grace_sec
        timeout = int(task.get("stall_timeout_sec", self.stall_timeout_sec))

        if time.time() - started_at < grace:
            return

        if elapsed_since_progress > timeout:
            phase_info = self._get_phase_info(task)
            if phase_info is None:
                return
            self._task_log(
                task,
                f"No progress for {int(elapsed_since_progress)}s in phase={state.get('phase')}; treating as stall",
            )
            self._handle_stall(task, state, phase_info)

    def _update_gpu_idle_rounds(
        self, gpu_process_map: Dict[int, List[str]], reserved_gpu_indices: List[int]
    ) -> None:
        reserved = set(int(index) for index in reserved_gpu_indices)
        all_gpu_indices = self._all_gpu_indices(gpu_process_map)

        for gpu_index in all_gpu_indices:
            busy = gpu_index in reserved or bool(gpu_process_map.get(gpu_index, []))
            if busy:
                self.gpu_idle_rounds[gpu_index] = 0
            else:
                self.gpu_idle_rounds[gpu_index] = self.gpu_idle_rounds.get(gpu_index, 0) + 1

        for gpu_index in list(self.gpu_idle_rounds):
            if gpu_index not in all_gpu_indices:
                self.gpu_idle_rounds.pop(gpu_index, None)

    def _select_gpu_indices_for_task(
        self,
        task: Dict,
        gpu_process_map: Dict[int, List[str]],
        reserved_gpu_indices: List[int],
        gpu_stats_map: Dict[int, Dict[str, float]],
    ) -> Optional[List[int]]:
        all_gpu_indices = sorted(set(self._all_gpu_indices(gpu_process_map)) | set(gpu_stats_map.keys()))
        if not all_gpu_indices:
            return None

        reserved = set(int(index) for index in reserved_gpu_indices)
        allowed_gpu_indices = [
            index
            for index in self._task_allowed_gpu_indices(task, all_gpu_indices)
            if index in all_gpu_indices
        ]
        if not allowed_gpu_indices:
            return None

        if self._task_schedule_mode(task) == "fixed":
            if self._fixed_partial_quick_gpu_ready(
                task,
                allowed_gpu_indices,
                gpu_process_map,
                gpu_stats_map,
                reserved_gpu_indices,
            ):
                return allowed_gpu_indices
            if self._gpu_set_busy(allowed_gpu_indices, gpu_process_map):
                return None
            if any(index in reserved for index in allowed_gpu_indices):
                return None
            if any(
                self.gpu_idle_rounds.get(index, 0) < self.gpu_idle_confirm_rounds
                for index in allowed_gpu_indices
            ):
                return None
            return allowed_gpu_indices

        gpu_count = self._task_requested_gpu_count(task)
        dynamic_partial_selection = self._dynamic_partial_quick_gpu_selection(
            task,
            allowed_gpu_indices,
            gpu_process_map,
            gpu_stats_map,
            reserved_gpu_indices,
        )
        if dynamic_partial_selection is not None:
            return dynamic_partial_selection

        available_gpu_indices = [
            index
            for index in allowed_gpu_indices
            if not gpu_process_map.get(index, [])
            and index not in reserved
            and self.gpu_idle_rounds.get(index, 0) >= self.gpu_idle_confirm_rounds
        ]
        if len(available_gpu_indices) < gpu_count:
            return None
        return available_gpu_indices[:gpu_count]

    def _write_status_snapshot(
        self,
        gpu_process_map: Dict[int, List[str]],
        gpu_stats_map: Dict[int, Dict[str, float]],
    ) -> None:
        all_gpu_indices = sorted(set(self._all_gpu_indices(gpu_process_map)) | set(gpu_stats_map.keys()))
        snapshot = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_process_map": gpu_process_map,
            "gpu_stats_map": gpu_stats_map,
            "tasks": [],
        }

        for task in self.tasks:
            state = self._load_task_state(task)
            phase_info = self._get_phase_info(task)
            assigned_gpu_indices = (
                self._assigned_gpu_indices(task, state) if state.get("pid") else []
            )
            snapshot["tasks"].append(
                {
                    "name": task["name"],
                    "schedule_mode": self._task_schedule_mode(task),
                    "requested_gpu_count": self._task_requested_gpu_count(task),
                    "configured_gpu_indices": self._task_fixed_gpu_indices(task),
                    "allowed_gpu_indices": self._task_allowed_gpu_indices(task, all_gpu_indices),
                    "assigned_gpu_indices": assigned_gpu_indices,
                    "gpu_indices": assigned_gpu_indices,
                    "output_dir": task["output_dir"],
                    "state": state,
                    "skip_problem_ids": self._read_skip_problem_ids(task),
                    "next_phase": phase_info.name if phase_info else None,
                    "next_problem_ids": phase_info.pending_problem_ids if phase_info else [],
                }
            )

        with (self.state_dir / "controller_status.json").open("w") as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)

    def run(self) -> None:
        self._log(f"Starting task controller with config {self.config_path}")
        while True:
            gpu_process_map = self._gpu_process_map()
            gpu_stats_map = self._gpu_stats_map()

            for task in self.tasks:
                self._reconcile_task_processes(task)

            for task in self.tasks:
                self._monitor_running_task(task)

            reserved_gpu_indices: List[int] = []
            for task in self.tasks:
                state = self._load_task_state(task)
                if state.get("pid"):
                    reserved_gpu_indices.extend(self._assigned_gpu_indices(task, state))

            self._update_gpu_idle_rounds(gpu_process_map, reserved_gpu_indices)

            for task in self.tasks:
                state = self._load_task_state(task)
                if state.get("pid"):
                    continue

                phase_info = self._get_phase_info(task)
                if phase_info is None:
                    continue

                cooldown_until = float(state.get("cooldown_until", 0))
                if time.time() < cooldown_until:
                    continue

                selected_gpu_indices = self._select_gpu_indices_for_task(
                    task, gpu_process_map, reserved_gpu_indices, gpu_stats_map
                )
                if not selected_gpu_indices:
                    continue

                self._start_task(task, phase_info, selected_gpu_indices)
                reserved_gpu_indices.extend(selected_gpu_indices)

            self._write_status_snapshot(gpu_process_map, gpu_stats_map)
            time.sleep(self.poll_interval_sec)


def main():
    if len(sys.argv) != 2:
        print("Usage: task_controller.py <config_path>", file=sys.stderr)
        sys.exit(2)

    controller = TaskController(sys.argv[1])
    controller.run()


if __name__ == "__main__":
    main()

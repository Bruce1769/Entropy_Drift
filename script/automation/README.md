# R2R Task Controller

This controller watches the server GPU state and launches each evaluation script when it can reserve enough idle GPUs.

Current task mapping in `resource/aime26_task_controller_config.json`:

- `aime26_entropy` -> any 2 idle GPUs
- `aime26_r2r_neural` -> any 2 idle GPUs
- `aime26_topk_llm` -> any 2 idle GPUs

Current behavior:

- each task requests `gpu_count=2`
- if at least 2 GPUs are idle, the controller assigns them automatically
- if not enough GPUs are idle, the task waits
- after the main pass finishes, problems with `has_extracted_answer=False` are rerun with a larger `MAX_NEW_TOKENS`
- skipped hard problems are persisted in `skip_problem_ids.txt`
- if a task stalls, the controller skips the current candidate problem and retries later

## Manual Usage

Start in background:

```bash
bash /remote-home/pxl/R2R/script/automation/run_task_controller.sh
```

Start in foreground:

```bash
bash /remote-home/pxl/R2R/script/automation/run_task_controller_foreground.sh
```

Stop:

```bash
bash /remote-home/pxl/R2R/script/automation/stop_task_controller.sh
```

Show status:

```bash
python3 /remote-home/pxl/R2R/script/automation/show_task_controller_status.py
```

Follow logs:

```bash
tail -f /remote-home/pxl/R2R/output/task_controller_state/logs/controller.log
tail -f /remote-home/pxl/R2R/output/task_controller_state/logs/aime26_entropy.log
tail -f /remote-home/pxl/R2R/output/task_controller_state/logs/aime26_r2r_neural.log
tail -f /remote-home/pxl/R2R/output/task_controller_state/logs/aime26_topk_llm.log
```

## State Files

- controller status: `output/task_controller_state/controller_status.json`
- controller pid: `output/task_controller_state/task_controller.pid`
- per-task state: `output/task_controller_state/tasks/<task_name>/state.json`
- per-task skip list: `output/task_controller_state/tasks/<task_name>/skip_problem_ids.txt`

## systemd

This container is not booted with `systemd` as PID 1, so service installation must be done on the host machine.

Prepared files:

- service template: `script/automation/r2r-task-controller.service`
- installer: `script/automation/install_task_controller_systemd.sh`

Host-side install command:

```bash
sudo bash /remote-home/pxl/R2R/script/automation/install_task_controller_systemd.sh
```

Useful host-side commands:

```bash
sudo systemctl status r2r-task-controller.service --no-pager
sudo journalctl -u r2r-task-controller.service -f
sudo systemctl restart r2r-task-controller.service
sudo systemctl stop r2r-task-controller.service
```

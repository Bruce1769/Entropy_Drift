import argparse
import csv
import itertools
import json
import os
import subprocess
from pathlib import Path


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep entropy_drift router knobs.")
    parser.add_argument("--problem-ids", default="1,2", help="Comma-separated AIME26 problem ids.")
    parser.add_argument("--cuda-visible-devices", default="6,7")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--rerun-max-new-tokens", type=int, default=8192)
    parser.add_argument("--alpha-values", default="0.3,0.5")
    parser.add_argument("--bias-values", default="0.3,0.5,0.7")
    parser.add_argument("--tau-values", default="0.1,0.2")
    parser.add_argument("--warmup-values", default="32,64")
    parser.add_argument("--hysteresis", type=float, default=0.15)
    parser.add_argument("--hold-tokens", type=int, default=4)
    parser.add_argument("--max-confident-prob", type=float, default=0.85)
    parser.add_argument("--output-root", default="output/eval/entropy_drift_sweeps")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--limit", type=int, default=6, help="Optional limit on the number of configs to run.")
    return parser.parse_args()


def make_tag(config: dict) -> str:
    parts = []
    for key in ("alpha", "bias", "tau", "warmup_steps"):
        value = str(config[key]).replace(".", "p")
        short_key = {"alpha": "a", "bias": "b", "tau": "t", "warmup_steps": "w"}[key]
        parts.append(f"{short_key}{value}")
    return "_".join(parts)


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[2]
    output_root = root_dir / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    configs = []
    if args.include_baseline:
        configs.append(
            {
                "alpha": 0.1,
                "bias": 0.0,
                "tau": 0.5,
                "warmup_steps": 10,
                "hysteresis": 0.0,
                "hold_tokens": 0,
                "max_confident_prob": 1.0,
                "stochastic": "1",
            }
        )

    for alpha, bias, tau, warmup_steps in itertools.product(
        parse_float_list(args.alpha_values),
        parse_float_list(args.bias_values),
        parse_float_list(args.tau_values),
        parse_int_list(args.warmup_values),
    ):
        configs.append(
            {
                "alpha": alpha,
                "bias": bias,
                "tau": tau,
                "warmup_steps": warmup_steps,
                "hysteresis": args.hysteresis,
                "hold_tokens": args.hold_tokens,
                "max_confident_prob": args.max_confident_prob,
                "stochastic": "0",
            }
        )

    if args.limit > 0:
        configs = configs[: args.limit]

    summary_rows = []
    for index, config in enumerate(configs, start=1):
        tag = make_tag(config)
        run_output_dir = output_root / f"{index:02d}_{tag}"
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
                "PROBLEM_IDS": args.problem_ids,
                "MAX_NEW_TOKENS": str(args.max_new_tokens),
                "RERUN_MAX_NEW_TOKENS": str(args.rerun_max_new_tokens),
                "RERUN_MISSING_ANSWERS": "1",
                "ALPHA": str(config["alpha"]),
                "BIAS": str(config["bias"]),
                "TAU": str(config["tau"]),
                "WARMUP_STEPS": str(config["warmup_steps"]),
                "HYSTERESIS": str(config["hysteresis"]),
                "HOLD_TOKENS": str(config["hold_tokens"]),
                "MAX_CONFIDENT_PROB": str(config["max_confident_prob"]),
                "STOCHASTIC": str(config["stochastic"]),
                "OUTPUT_DIR": str(run_output_dir),
            }
        )
        subprocess.run(
            ["bash", "script/evaluate/run_qwen3_0_6b_qwen3_8b_aime26_entropy_drift.sh"],
            cwd=root_dir,
            env=env,
            check=True,
        )
        analysis_prefix = run_output_dir / "analysis" / "drift_metrics"
        subprocess.run(
            [
                "python",
                "script/evaluate/analyze_entropy_drift_token_traces.py",
                "--input",
                str(run_output_dir / "token_traces"),
                "--output-prefix",
                str(analysis_prefix),
                "--bias",
                str(config["bias"]),
            ],
            cwd=root_dir,
            env=env,
            check=True,
        )
        with open(f"{analysis_prefix}_summary.json", "r") as f:
            summary = json.load(f)
        summary_rows.append(
            {
                "tag": tag,
                "output_dir": str(run_output_dir),
                "alpha": config["alpha"],
                "bias": config["bias"],
                "tau": config["tau"],
                "warmup_steps": config["warmup_steps"],
                "hysteresis": config["hysteresis"],
                "hold_tokens": config["hold_tokens"],
                "max_confident_prob": config["max_confident_prob"],
                "stochastic": config["stochastic"],
                "avg_reference_eval_ratio": summary["avg_reference_eval_ratio"],
                "avg_flip_rate": summary["avg_flip_rate"],
                "avg_false_positive_rate": summary["avg_false_positive_rate"],
                "avg_observed_true_positive_rate": summary["avg_observed_true_positive_rate"],
                "avg_below_zero_routed_rate": summary["avg_below_zero_routed_rate"],
                "avg_below_bias_routed_rate": summary["avg_below_bias_routed_rate"],
            }
        )

    summary_path = output_root / "sweep_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote sweep summary to {summary_path}")


if __name__ == "__main__":
    main()

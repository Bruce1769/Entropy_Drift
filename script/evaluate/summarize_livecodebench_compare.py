import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def load_stats(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "metric_name" not in df.columns or "value" not in df.columns:
        return {}
    return {
        str(row["metric_name"]): float(row["value"])
        for _, row in df.iterrows()
        if pd.notna(row["value"])
    }


def get_metrics_file(run_dir: Path) -> Optional[Path]:
    evaluation_file = run_dir / "combined_results_evaluation_light.csv"
    if evaluation_file.exists():
        return evaluation_file

    combined_file = run_dir / "combined_results.csv"
    if combined_file.exists():
        return combined_file

    return None


def maybe_mean(df: pd.DataFrame, column: str) -> Optional[float]:
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def maybe_rate(df: pd.DataFrame, column: str) -> Optional[float]:
    value = maybe_mean(df, column)
    if value is None:
        return None
    return value * 100.0


def compute_speed_tokens_per_second(df: pd.DataFrame, stats: Dict[str, float]) -> Optional[float]:
    if "speed_tokens_per_second" in stats:
        return float(stats["speed_tokens_per_second"])

    if "output_tokens" not in df.columns or "run_time" not in df.columns:
        return None

    output_tokens = pd.to_numeric(df["output_tokens"], errors="coerce")
    run_time = pd.to_numeric(df["run_time"], errors="coerce")
    valid = output_tokens.notna() & run_time.notna() & (run_time > 0)
    if not valid.any():
        return None

    total_output_tokens = float(output_tokens[valid].sum())
    total_run_time = float(run_time[valid].sum())
    if total_run_time <= 0:
        return None
    return total_output_tokens / total_run_time


def summarize_run(method: str, run_dir: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "method": method,
        "output_dir": str(run_dir),
        "status": "missing",
    }

    metrics_file = get_metrics_file(run_dir)
    if metrics_file is None:
        return row

    df = pd.read_csv(metrics_file)
    args = load_json(run_dir / "args.json")
    model_config = load_json(run_dir / "model_config.json")
    stats = load_stats(run_dir / "stats.csv")

    router_config = model_config.get("router", {})
    switching_strategy = router_config.get("switching_strategy")
    if switching_strategy is None and method == "r2r_neural":
        switching_strategy = "neural"

    row.update(
        {
            "status": "ok",
            "dataset": args.get("dataset"),
            "config_path": args.get("config_path"),
            "switching_strategy": switching_strategy,
            "metric_file": metrics_file.name,
            "num_samples": int(len(df)),
            "num_problems": int(df["problem_id"].nunique()) if "problem_id" in df.columns else int(len(df)),
            "pass_at_1": maybe_rate(df, "is_correct"),
            "extraction_rate": maybe_rate(df, "has_extracted_answer"),
            "avg_input_tokens": maybe_mean(df, "input_tokens"),
            "avg_output_tokens": maybe_mean(df, "output_tokens"),
            "avg_total_tokens": maybe_mean(df, "total_tokens"),
            "quick_usage_pct": maybe_mean(df, "quick_model_percentage"),
            "reference_usage_pct": maybe_mean(df, "reference_model_percentage"),
            "avg_params_billions": maybe_mean(df, "avg_params_billions"),
            "speed_tokens_per_second": compute_speed_tokens_per_second(df, stats),
        }
    )
    return row


def render_markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "| method | status |\n|---|---|\n"

    display_df = df.loc[:, columns].copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )
        else:
            display_df[column] = display_df[column].fillna("")

    header = "| " + " | ".join(display_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display_df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator] + rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize LiveCodeBench comparison runs.")
    parser.add_argument("--output_root", type=str, required=True, help="Parent directory containing per-method run folders.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["r2r_neural", "entropy", "topk_llm", "llm"],
        help="Method subdirectories to summarize.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows = [summarize_run(method, output_root / method) for method in args.methods]
    summary_df = pd.DataFrame(rows)

    output_csv = output_root / "livecodebench_comparison_summary.csv"
    output_md = output_root / "livecodebench_comparison_summary.md"

    summary_df.to_csv(output_csv, index=False)

    markdown_columns = [
        "method",
        "status",
        "switching_strategy",
        "pass_at_1",
        "extraction_rate",
        "avg_total_tokens",
        "speed_tokens_per_second",
        "quick_usage_pct",
        "reference_usage_pct",
    ]
    output_md.write_text(render_markdown_table(summary_df, markdown_columns))

    print(f"Saved summary CSV to {output_csv}")
    print(f"Saved summary Markdown to {output_md}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

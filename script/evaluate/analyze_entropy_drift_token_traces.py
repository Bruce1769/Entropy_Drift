import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize entropy_drift waste metrics from token_traces CSV files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A token_traces directory, a single CSV, or a glob pattern.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Writes `<prefix>.csv` and `<prefix>_summary.json`.",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="Optional drift bias used to report below-bias routed positions.",
    )
    return parser.parse_args()


def resolve_inputs(input_arg: str) -> list[Path]:
    path = Path(input_arg)
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    if path.is_file():
        return [path]
    return sorted(Path().glob(input_arg))


def compute_problem_metrics(df: pd.DataFrame, bias: float) -> dict:
    df = df.copy()
    df["router_decision"] = df["router_decision"].fillna(0).astype(int)
    df["reference_available"] = df["reference_token_id"].notna()
    df["top1_agree"] = (
        df["reference_available"]
        & (df["quick_token_id"].astype("Int64") == df["reference_token_id"].astype("Int64"))
    )
    df["top1_disagree"] = (
        df["reference_available"]
        & (df["quick_token_id"].astype("Int64") != df["reference_token_id"].astype("Int64"))
    )
    routed_df = df[df["router_decision"] == 1]
    routed_with_ref = routed_df[routed_df["reference_available"]]

    flips = int((df["router_decision"].diff().fillna(0) != 0).sum())
    flip_denominator = max(len(df) - 1, 1)
    fully_covered = bool(df["reference_available"].all())

    metrics = {
        "problem_id": int(df["problem_id"].iloc[0]),
        "positions": int(len(df)),
        "reference_eval_ratio": float(df["router_decision"].mean()),
        "switch_count": flips,
        "flip_rate": float(flips / flip_denominator),
        "routed_positions": int(len(routed_df)),
        "routed_positions_with_reference": int(len(routed_with_ref)),
        "false_positive_count": int(routed_with_ref["top1_agree"].sum()),
        "false_positive_rate": float(routed_with_ref["top1_agree"].mean()) if len(routed_with_ref) else 0.0,
        "observed_true_positive_count": int(routed_with_ref["top1_disagree"].sum()),
        "observed_true_positive_rate": float(routed_with_ref["top1_disagree"].mean()) if len(routed_with_ref) else 0.0,
        "below_zero_routed_rate": float((routed_df["router_score"] < 0).mean()) if len(routed_df) else 0.0,
        "below_bias_routed_rate": float((routed_df["router_score"] < bias).mean()) if len(routed_df) else 0.0,
        "full_reference_coverage": fully_covered,
        "disagreement_recall": None,
    }

    if fully_covered:
        disagreement_mask = df["top1_disagree"]
        disagreement_total = int(disagreement_mask.sum())
        recall = (
            float(df.loc[disagreement_mask, "router_decision"].mean())
            if disagreement_total
            else 0.0
        )
        metrics["disagreement_recall"] = recall
        metrics["disagreement_total"] = disagreement_total

    return metrics


def aggregate_metrics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    summary = {
        "num_problems": len(rows),
        "avg_reference_eval_ratio": 0.0,
        "avg_flip_rate": 0.0,
        "avg_false_positive_rate": 0.0,
        "avg_observed_true_positive_rate": 0.0,
        "avg_below_zero_routed_rate": 0.0,
        "avg_below_bias_routed_rate": 0.0,
        "full_reference_coverage_problems": 0,
        "avg_disagreement_recall": None,
    }
    if not rows:
        return summary

    for key in (
        "avg_reference_eval_ratio",
        "avg_flip_rate",
        "avg_false_positive_rate",
        "avg_observed_true_positive_rate",
        "avg_below_zero_routed_rate",
        "avg_below_bias_routed_rate",
    ):
        source_key = key.replace("avg_", "")
        summary[key] = float(sum(row[source_key] for row in rows) / len(rows))

    covered_rows = [row for row in rows if row["full_reference_coverage"]]
    summary["full_reference_coverage_problems"] = len(covered_rows)
    if covered_rows:
        summary["avg_disagreement_recall"] = float(
            sum(row["disagreement_recall"] for row in covered_rows) / len(covered_rows)
        )
    return summary


def main():
    args = parse_args()
    paths = resolve_inputs(args.input)
    if not paths:
        raise FileNotFoundError(f"No token trace CSV files found for input: {args.input}")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        rows.append(compute_problem_metrics(df, bias=args.bias))

    metrics_df = pd.DataFrame(rows).sort_values("problem_id")
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(f"{output_prefix}.csv", index=False)

    summary = aggregate_metrics(rows)
    summary["inputs"] = [str(path) for path in paths]
    with open(f"{output_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

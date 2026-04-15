import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize overlap between replayed router decisions and disagreement signals "
            "such as JS, entropy, and top1 mismatch."
        )
    )
    parser.add_argument("--comparison", required=True, help="Replay comparison csv.")
    parser.add_argument("--output-prefix", required=True, help="Prefix for csv/json outputs.")
    parser.add_argument(
        "--router-col",
        default="router_decision_replayed",
        help="Binary router decision column: 1=route to reference, 0=stay on quick.",
    )
    parser.add_argument("--js-col", default="js", help="JS divergence column.")
    parser.add_argument("--entropy-col", default="slm_entropy", help="SLM entropy column.")
    parser.add_argument("--top1-same-col", default="top1_same", help="Whether SLM and LLM top1 agree.")
    parser.add_argument(
        "--js-thresholds",
        default="0.05,0.1,0.2,0.3,0.5",
        help="Comma-separated JS thresholds for high-JS overlap analysis.",
    )
    parser.add_argument(
        "--entropy-thresholds",
        default="0.1,0.25,0.5,1.0,2.0",
        help="Comma-separated entropy thresholds for high-entropy overlap analysis.",
    )
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def summarize_series(series: pd.Series) -> dict:
    clean = series.dropna()
    if clean.empty:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p90": float(clean.quantile(0.9)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def compute_binary_overlap_stats(df: pd.DataFrame, label_col: str, pred_col: str) -> dict:
    label = df[label_col].astype(bool)
    pred = df[pred_col].astype(bool)
    tp = int((label & pred).sum())
    fp = int((~label & pred).sum())
    fn = int((label & ~pred).sum())
    tn = int((~label & ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.comparison).copy()

    required = [args.router_col, args.js_col, args.entropy_col, args.top1_same_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[args.router_col] = df[args.router_col].astype(int)
    df["top1_diff"] = ~df[args.top1_same_col].astype(bool)
    df["router_on"] = df[args.router_col] == 1
    df["router_off"] = df[args.router_col] == 0

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "positions_total": int(len(df)),
        "router_on_count": int(df["router_on"].sum()),
        "router_off_count": int(df["router_off"].sum()),
        "top1_diff_count": int(df["top1_diff"].sum()),
        "router_on_top1_diff_overlap": int((df["router_on"] & df["top1_diff"]).sum()),
        "router_off_top1_diff_count": int((df["router_off"] & df["top1_diff"]).sum()),
        "js_router_on": summarize_series(df.loc[df["router_on"], args.js_col]),
        "js_router_off": summarize_series(df.loc[df["router_off"], args.js_col]),
        "entropy_router_on": summarize_series(df.loc[df["router_on"], args.entropy_col]),
        "entropy_router_off": summarize_series(df.loc[df["router_off"], args.entropy_col]),
        "js_top1_diff": summarize_series(df.loc[df["top1_diff"], args.js_col]),
        "js_top1_same": summarize_series(df.loc[~df["top1_diff"], args.js_col]),
        "entropy_top1_diff": summarize_series(df.loc[df["top1_diff"], args.entropy_col]),
        "entropy_top1_same": summarize_series(df.loc[~df["top1_diff"], args.entropy_col]),
    }

    threshold_rows = []
    for threshold in parse_thresholds(args.js_thresholds):
        pred_col = f"js_ge_{threshold}"
        df[pred_col] = df[args.js_col] >= threshold
        stats_vs_router = compute_binary_overlap_stats(df, "router_on", pred_col)
        stats_vs_top1 = compute_binary_overlap_stats(df, "top1_diff", pred_col)
        threshold_rows.append(
            {
                "signal": "js",
                "threshold": threshold,
                "positive_count": int(df[pred_col].sum()),
                "vs_router_precision": stats_vs_router["precision"],
                "vs_router_recall": stats_vs_router["recall"],
                "vs_router_f1": stats_vs_router["f1"],
                "vs_top1diff_precision": stats_vs_top1["precision"],
                "vs_top1diff_recall": stats_vs_top1["recall"],
                "vs_top1diff_f1": stats_vs_top1["f1"],
            }
        )

    for threshold in parse_thresholds(args.entropy_thresholds):
        pred_col = f"entropy_ge_{threshold}"
        df[pred_col] = df[args.entropy_col] >= threshold
        stats_vs_router = compute_binary_overlap_stats(df, "router_on", pred_col)
        stats_vs_top1 = compute_binary_overlap_stats(df, "top1_diff", pred_col)
        threshold_rows.append(
            {
                "signal": "entropy",
                "threshold": threshold,
                "positive_count": int(df[pred_col].sum()),
                "vs_router_precision": stats_vs_router["precision"],
                "vs_router_recall": stats_vs_router["recall"],
                "vs_router_f1": stats_vs_router["f1"],
                "vs_top1diff_precision": stats_vs_top1["precision"],
                "vs_top1diff_recall": stats_vs_top1["recall"],
                "vs_top1diff_f1": stats_vs_top1["f1"],
            }
        )

    pd.DataFrame(threshold_rows).to_csv(f"{output_prefix}_thresholds.csv", index=False)
    with open(f"{output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote summary to {output_prefix}_summary.json")
    print(f"Wrote threshold sweep to {output_prefix}_thresholds.csv")


if __name__ == "__main__":
    main()

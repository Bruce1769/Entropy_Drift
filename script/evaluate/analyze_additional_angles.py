import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze additional angles beyond entropy/JS on replay comparison results."
    )
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--js-threshold", type=float, default=0.1)
    parser.add_argument("--tail-window", type=int, default=32)
    return parser.parse_args()


def summarize(series: pd.Series) -> dict:
    series = series.dropna()
    if series.empty:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.9)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def classify_token(token: str) -> str:
    token = "" if pd.isna(token) else str(token)
    stripped = token.strip()
    if "<|" in token or token.startswith("<") or token.endswith(">"):
        return "special"
    if stripped == "":
        return "whitespace"
    if re.fullmatch(r"[0-9]+", stripped):
        return "digits"
    if re.fullmatch(r"[\W_]+", stripped):
        return "punctuation"
    if any(ch.isdigit() for ch in stripped):
        return "mixed_digits"
    if re.search(r"[=+\-*/^\\{}()\[\]]", stripped):
        return "math_symbolic"
    return "text"


def compute_span_summary(mask: pd.Series) -> dict:
    lengths = []
    current = 0
    for value in mask.astype(bool).tolist():
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    if not lengths:
        return {"span_count": 0, "mean_len": None, "median_len": None, "p90_len": None, "max_len": None}
    series = pd.Series(lengths, dtype=float)
    return {
        "span_count": int(len(lengths)),
        "mean_len": float(series.mean()),
        "median_len": float(series.median()),
        "p90_len": float(series.quantile(0.9)),
        "max_len": int(series.max()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.comparison).copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["top1_same"] = df["top1_same"].astype(bool)
    df["router_on"] = df["router_decision_replayed"] == 1
    df["top1_diff"] = ~df["top1_same"]
    df["js_high"] = df["js"] > args.js_threshold
    df["token_type"] = df["output_token_str"].apply(classify_token)
    df["slm_top1_type"] = df["slm_top1_str"].apply(classify_token)
    df["llm_top1_type"] = df["llm_top1_str"].apply(classify_token)

    sample_lengths = df.groupby("sample_id")["position"].max().add(1).rename("sample_len")
    df = df.merge(sample_lengths, on="sample_id", how="left")
    df["rel_pos"] = (df["position"] + 0.5) / df["sample_len"]
    df["pos_decile"] = pd.cut(
        df["rel_pos"],
        bins=[i / 10 for i in range(11)],
        labels=[f"d{i}" for i in range(1, 11)],
        include_lowest=True,
    )

    # 1) Top-k/rank-shift proxy via token-type divergence.
    proxy_summary = {
        "note": (
            "True top-k overlap and rank-shift are unavailable from the current comparison file, "
            "because it stores only top1 ids/strings, entropy, JS, and router decision, not full "
            "logits/top-k ranks. The tables below are a weaker proxy based on top1 token categories."
        ),
        "slm_top1_type_on_top1diff": df[df["top1_diff"]]["slm_top1_type"].value_counts(normalize=True).to_dict(),
        "llm_top1_type_on_top1diff": df[df["top1_diff"]]["llm_top1_type"].value_counts(normalize=True).to_dict(),
        "router0_top1diff_slm_type": df[(~df["router_on"]) & (df["top1_diff"])]["slm_top1_type"].value_counts(normalize=True).to_dict(),
        "router1_top1diff_slm_type": df[(df["router_on"]) & (df["top1_diff"])]["slm_top1_type"].value_counts(normalize=True).to_dict(),
    }
    with open(output_dir / "topk_rank_proxy_summary.json", "w", encoding="utf-8") as f:
        json.dump(proxy_summary, f, indent=2, ensure_ascii=False)

    # 2) Position-wise dynamics.
    position_dynamics = (
        df.groupby("pos_decile", observed=False)
        .agg(
            count=("sample_id", "size"),
            router_rate=("router_on", "mean"),
            top1_diff_rate=("top1_diff", "mean"),
            js_mean=("js", "mean"),
            entropy_mean=("slm_entropy", "mean"),
        )
        .reset_index()
    )
    position_dynamics.to_csv(output_dir / "position_dynamics.csv", index=False)

    # 3) Local burst / contiguous spans.
    span_rows = []
    for sample_id, sample_df in df.sort_values(["sample_id", "position"]).groupby("sample_id"):
        span_rows.append({"sample_id": sample_id, "signal": "router_on", **compute_span_summary(sample_df["router_on"])})
        span_rows.append({"sample_id": sample_id, "signal": "top1_diff", **compute_span_summary(sample_df["top1_diff"])})
        span_rows.append({"sample_id": sample_id, "signal": "js_high", **compute_span_summary(sample_df["js_high"])})
    span_df = pd.DataFrame(span_rows)
    span_df.to_csv(output_dir / "span_stats_by_sample.csv", index=False)
    span_summary = (
        span_df.groupby("signal")
        .agg(
            samples=("sample_id", "size"),
            avg_span_count=("span_count", "mean"),
            avg_mean_len=("mean_len", "mean"),
            avg_p90_len=("p90_len", "mean"),
            max_len=("max_len", "max"),
        )
        .reset_index()
    )
    span_summary.to_csv(output_dir / "span_stats_summary.csv", index=False)

    # 4) Router miss / false alarm.
    miss_df = df[(~df["router_on"]) & (df["top1_diff"])].copy()
    false_alarm_df = df[(df["router_on"]) & (~df["top1_diff"])].copy()
    miss_false_summary = {
        "router_miss": {
            "count": int(len(miss_df)),
            "js": summarize(miss_df["js"]),
            "entropy": summarize(miss_df["slm_entropy"]),
            "position_deciles": miss_df["pos_decile"].value_counts(normalize=True).sort_index().to_dict(),
            "token_types": miss_df["token_type"].value_counts(normalize=True).to_dict(),
        },
        "router_false_alarm": {
            "count": int(len(false_alarm_df)),
            "js": summarize(false_alarm_df["js"]),
            "entropy": summarize(false_alarm_df["slm_entropy"]),
            "position_deciles": false_alarm_df["pos_decile"].value_counts(normalize=True).sort_index().to_dict(),
            "token_types": false_alarm_df["token_type"].value_counts(normalize=True).to_dict(),
        },
    }
    with open(output_dir / "miss_false_alarm_summary.json", "w", encoding="utf-8") as f:
        json.dump(miss_false_summary, f, indent=2, ensure_ascii=False)

    # 5) Coverage-efficiency tradeoff.
    threshold_rows = []
    total = len(df)
    total_diff = int(df["top1_diff"].sum())
    for signal in ("js", "slm_entropy"):
        for quantile in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
            threshold = float(df[signal].quantile(quantile))
            selected = df[signal] >= threshold
            selected_count = int(selected.sum())
            hit = int((selected & df["top1_diff"]).sum())
            precision = hit / selected_count if selected_count else None
            recall = hit / total_diff if total_diff else None
            threshold_rows.append(
                {
                    "signal": signal,
                    "quantile": quantile,
                    "threshold": threshold,
                    "selected_count": selected_count,
                    "selected_ratio": selected_count / total,
                    "top1diff_hit_count": hit,
                    "top1diff_precision": precision,
                    "top1diff_recall": recall,
                }
            )
    tradeoff_df = pd.DataFrame(threshold_rows)
    tradeoff_df.to_csv(output_dir / "coverage_efficiency_tradeoff.csv", index=False)

    # 6) Answer-critical positions heuristic.
    df["tail32"] = (df["sample_len"] - df["position"]) <= args.tail_window
    answer_critical_summary = {
        "tail_window": args.tail_window,
        "tail32": {
            "count": int(df["tail32"].sum()),
            "router_rate": float(df.loc[df["tail32"], "router_on"].mean()),
            "top1_diff_rate": float(df.loc[df["tail32"], "top1_diff"].mean()),
            "js": summarize(df.loc[df["tail32"], "js"]),
            "entropy": summarize(df.loc[df["tail32"], "slm_entropy"]),
        },
        "non_tail32": {
            "count": int((~df["tail32"]).sum()),
            "router_rate": float(df.loc[~df["tail32"], "router_on"].mean()),
            "top1_diff_rate": float(df.loc[~df["tail32"], "top1_diff"].mean()),
            "js": summarize(df.loc[~df["tail32"], "js"]),
            "entropy": summarize(df.loc[~df["tail32"], "slm_entropy"]),
        },
        "digit_tokens_tail32": {
            "count": int(((df["tail32"]) & (df["token_type"].isin(["digits", "mixed_digits"]))).sum()),
            "top1_diff_rate": float(
                df.loc[(df["tail32"]) & (df["token_type"].isin(["digits", "mixed_digits"])), "top1_diff"].mean()
            ),
            "router_rate": float(
                df.loc[(df["tail32"]) & (df["token_type"].isin(["digits", "mixed_digits"])), "router_on"].mean()
            ),
        },
    }
    with open(output_dir / "answer_critical_summary.json", "w", encoding="utf-8") as f:
        json.dump(answer_critical_summary, f, indent=2, ensure_ascii=False)

    final_summary = {
        "comparison_file": args.comparison,
        "js_threshold_for_js_high": args.js_threshold,
        "tail_window": args.tail_window,
        "generated_files": [
            "topk_rank_proxy_summary.json",
            "position_dynamics.csv",
            "span_stats_by_sample.csv",
            "span_stats_summary.csv",
            "miss_false_alarm_summary.json",
            "coverage_efficiency_tradeoff.csv",
            "answer_critical_summary.json",
        ],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote additional analyses to {output_dir}")


if __name__ == "__main__":
    main()

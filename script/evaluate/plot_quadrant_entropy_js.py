#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def add_quadrant(df: pd.DataFrame) -> pd.DataFrame:
    required = {"router_decision_replayed", "top1_same", "slm_entropy", "js"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["router_decision_replayed"] = df["router_decision_replayed"].astype(int)
    df["top1_same"] = df["top1_same"].astype(int)
    df["quadrant"] = np.where(
        df["router_decision_replayed"] == 1,
        np.where(df["top1_same"] == 1, "router1_top1same", "router1_top1diff"),
        np.where(df["top1_same"] == 1, "router0_top1same", "router0_top1diff"),
    )
    return df


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for quadrant, group in df.groupby("quadrant"):
        ent = group["slm_entropy"].to_numpy(dtype=float)
        js = group["js"].to_numpy(dtype=float)
        ent_stats = compute_stats(ent)
        js_stats = compute_stats(js)
        row = {
            "quadrant": quadrant,
            "entropy_count": ent_stats["count"],
            "entropy_mean": ent_stats["mean"],
            "entropy_median": ent_stats["median"],
            "entropy_p90": ent_stats["p90"],
            "entropy_p95": ent_stats["p95"],
            "entropy_min": ent_stats["min"],
            "entropy_max": ent_stats["max"],
            "js_count": js_stats["count"],
            "js_mean": js_stats["mean"],
            "js_median": js_stats["median"],
            "js_p90": js_stats["p90"],
            "js_p95": js_stats["p95"],
            "js_min": js_stats["min"],
            "js_max": js_stats["max"],
        }
        records.append(row)
    return pd.DataFrame(records).sort_values("quadrant")


def plot_distribution(df: pd.DataFrame, column: str, output_path: str, title: str) -> None:
    colors = {
        "router1_top1same": "#1f77b4",
        "router1_top1diff": "#d62728",
        "router0_top1same": "#2ca02c",
        "router0_top1diff": "#ff7f0e",
    }
    plt.figure(figsize=(10, 6))
    all_values = df[column].to_numpy(dtype=float)
    if column == "slm_entropy":
        bins = np.linspace(0.0, max(1.0, np.percentile(all_values, 99.5)), 80)
    else:
        bins = np.linspace(0.0, max(0.2, np.percentile(all_values, 99.5)), 80)
    for quadrant, group in df.groupby("quadrant"):
        values = group[column].to_numpy(dtype=float)
        label = f"{quadrant} (n={values.size})"
        plt.hist(values, bins=bins, density=True, alpha=0.45, label=label, color=colors.get(quadrant))
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot entropy/JS distributions by router/top1 quadrant.")
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to comparison CSV with router_decision_replayed, top1_same, slm_entropy, js columns.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write plots and stats.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    df = pd.read_csv(args.input_csv)
    df = add_quadrant(df)

    stats_df = summarize_quadrants(df)
    stats_csv = os.path.join(args.output_dir, "quadrant_stats.csv")
    stats_json = os.path.join(args.output_dir, "quadrant_stats.json")
    stats_df.to_csv(stats_csv, index=False)
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    plot_distribution(
        df,
        "slm_entropy",
        os.path.join(args.output_dir, "entropy_distribution.png"),
        "SLM Entropy Distribution by Quadrant",
    )
    plot_distribution(
        df,
        "js",
        os.path.join(args.output_dir, "js_distribution.png"),
        "JS Distribution by Quadrant",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


CASE_COLUMNS = [
    "problem_id",
    "position",
    "router_decision",
    "top1_same",
    "slm_entropy",
    "js",
    "output_token_str_online",
    "slm_top1_str",
    "llm_top1_str",
    "quick_token_str",
    "source_model",
]


def take_cases(sub: pd.DataFrame, per_bucket: int) -> pd.DataFrame:
    ordered = sub.sort_values(["js", "slm_entropy"], ascending=[False, False])
    buckets = [
        ordered.head(per_bucket),
        sub.sort_values("js", ascending=False).head(per_bucket),
        sub.sort_values("slm_entropy", ascending=False).head(per_bucket),
        sub[sub["top1_same"] == False].sort_values(["js", "slm_entropy"], ascending=[False, False]).head(per_bucket),
        sub[sub["top1_same"] == True].sort_values(["js", "slm_entropy"], ascending=[False, False]).head(per_bucket),
    ]
    picked = pd.concat([b for b in buckets if not b.empty], axis=0).drop_duplicates(subset=["problem_id", "position"])
    return picked.sort_values(["js", "slm_entropy"], ascending=[False, False]).head(per_bucket * 4)


def token_repr(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", "\\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export representative threshold-quadrant cases.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quadrant-col", default="threshold_quadrant_name")
    parser.add_argument("--per-quadrant", type=int, default=40)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df["quadrant"] = df[args.quadrant_col].astype(str)
    if "top1_same" in df.columns:
        df["top1_same"] = df["top1_same"].astype(str).str.lower().isin(["1", "true"])
    else:
        df["top1_same"] = df["slm_top1_str"].fillna("").astype(str) == df["llm_top1_str"].fillna("").astype(str)

    summary_rows = []
    report_lines = ["# Threshold Quadrant Case Study", ""]

    for quadrant, sub in df.groupby("quadrant"):
        cases = take_cases(sub, args.per_quadrant)
        out = cases[[c for c in CASE_COLUMNS if c in cases.columns]].copy()
        for col in ["output_token_str_online", "slm_top1_str", "llm_top1_str", "quick_token_str"]:
            if col in out.columns:
                out[col] = out[col].map(token_repr)
        out_path = output_dir / f"{quadrant}_cases.csv"
        out.to_csv(out_path, index=False)

        summary_rows.append(
            {
                "quadrant": quadrant,
                "count": int(len(sub)),
                "top1_same_rate": float(sub["top1_same"].mean()),
                "js_mean": float(sub["js"].mean()),
                "js_p90": float(sub["js"].quantile(0.9)),
                "entropy_mean": float(sub["slm_entropy"].mean()),
                "entropy_p90": float(sub["slm_entropy"].quantile(0.9)),
            }
        )

        top_tokens = (
            sub["output_token_str_online"]
            .fillna("")
            .astype(str)
            .str.replace("\n", "\\n", regex=False)
            .value_counts()
            .head(10)
        )
        report_lines.append(f"## {quadrant}")
        report_lines.append(f"- count={len(sub)}")
        report_lines.append(f"- top1_same_rate={sub['top1_same'].mean():.4f}")
        report_lines.append(f"- js_mean={sub['js'].mean():.4f}, js_p90={sub['js'].quantile(0.9):.4f}")
        report_lines.append(
            f"- entropy_mean={sub['slm_entropy'].mean():.4f}, entropy_p90={sub['slm_entropy'].quantile(0.9):.4f}"
        )
        report_lines.append("- top output tokens: " + ", ".join(f"{tok}({cnt})" for tok, cnt in top_tokens.items()))
        report_lines.append("")

    pd.DataFrame(summary_rows).to_csv(output_dir / "threshold_quadrant_case_summary.csv", index=False)
    (output_dir / "threshold_quadrant_case_study.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

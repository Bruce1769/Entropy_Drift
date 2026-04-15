import argparse
import ast
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze routed positions by comparing SLM/LLM top1, entropy, and JS "
            "from a per-position comparison table."
        )
    )
    parser.add_argument("--routes", required=True, help="Route positions file (.csv or .jsonl).")
    parser.add_argument(
        "--comparison",
        required=True,
        help="Per-position comparison file (.csv or .jsonl).",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for outputs. Generates `<prefix>.csv` and `<prefix>_summary.json`.",
    )
    parser.add_argument("--sample-id-col", default="sample_id", help="Sample id column shared by both inputs.")
    parser.add_argument("--position-col", default="position", help="Position column in comparison file.")
    parser.add_argument(
        "--route-positions-col",
        default="route_positions",
        help="Column in routes file containing routed position lists.",
    )
    parser.add_argument("--slm-top1-col", default="slm_top1_id", help="Optional precomputed SLM top1 column.")
    parser.add_argument("--llm-top1-col", default="llm_top1_id", help="Optional precomputed LLM top1 column.")
    parser.add_argument("--slm-entropy-col", default="slm_entropy", help="Optional precomputed SLM entropy column.")
    parser.add_argument("--js-col", default="js", help="Optional precomputed JS column.")
    parser.add_argument("--slm-logits-col", default="slm_logits", help="Optional full-logits column for SLM.")
    parser.add_argument("--llm-logits-col", default="llm_logits", help="Optional full-logits column for LLM.")
    parser.add_argument(
        "--slm-topk-indices-col",
        default="slm_topk_indices",
        help="Optional sparse top-k indices column for SLM.",
    )
    parser.add_argument(
        "--slm-topk-logits-col",
        default="slm_topk_logits",
        help="Optional sparse top-k logits column for SLM.",
    )
    parser.add_argument(
        "--llm-topk-indices-col",
        default="llm_topk_indices",
        help="Optional sparse top-k indices column for LLM.",
    )
    parser.add_argument(
        "--llm-topk-logits-col",
        default="llm_topk_logits",
        help="Optional sparse top-k logits column for LLM.",
    )
    return parser.parse_args()


def load_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file format: {path}")


def parse_maybe_list(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return value
    return value


def ensure_list_of_numbers(value: Any) -> Optional[list[float]]:
    parsed = parse_maybe_list(value)
    if parsed is None:
        return None
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list-like value, got: {type(parsed)}")
    return [float(x) for x in parsed]


def ensure_list_of_ints(value: Any) -> Optional[list[int]]:
    parsed = parse_maybe_list(value)
    if parsed is None:
        return None
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list-like value, got: {type(parsed)}")
    return [int(x) for x in parsed]


def compute_entropy_from_logits(logits: list[float]) -> float:
    tensor = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(tensor, dim=-1)
    log_probs = F.log_softmax(tensor, dim=-1)
    return float(-(probs * log_probs).sum().item())


def compute_js_from_logits(slm_logits: list[float], llm_logits: list[float]) -> float:
    p = torch.tensor(slm_logits, dtype=torch.float32)
    q = torch.tensor(llm_logits, dtype=torch.float32)
    log_p = F.log_softmax(p, dim=-1)
    log_q = F.log_softmax(q, dim=-1)
    prob_p = log_p.exp()
    prob_q = log_q.exp()
    mean_prob = 0.5 * (prob_p + prob_q)
    log_mean = torch.log(mean_prob.clamp_min(1e-12))
    kl_pm = (prob_p * (log_p - log_mean)).sum()
    kl_qm = (prob_q * (log_q - log_mean)).sum()
    return float((0.5 * (kl_pm + kl_qm)).item())


def compute_sparse_topk_js(
    slm_indices: list[int],
    slm_logits: list[float],
    llm_indices: list[int],
    llm_logits: list[float],
) -> float:
    p_idx = torch.tensor(slm_indices, dtype=torch.long)
    q_idx = torch.tensor(llm_indices, dtype=torch.long)
    p_logits = torch.tensor(slm_logits, dtype=torch.float32)
    q_logits = torch.tensor(llm_logits, dtype=torch.float32)

    combined = torch.cat([p_idx, q_idx], dim=0)
    union, inverse = torch.unique(combined, sorted=True, return_inverse=True)
    p_pos = inverse[: p_idx.shape[0]]
    q_pos = inverse[p_idx.shape[0] :]

    p_union = torch.full((union.shape[0],), float("-inf"), dtype=torch.float32)
    q_union = torch.full((union.shape[0],), float("-inf"), dtype=torch.float32)
    p_union[p_pos] = p_logits
    q_union[q_pos] = q_logits

    prob_p = torch.softmax(p_union, dim=-1)
    prob_q = torch.softmax(q_union, dim=-1)
    mean_prob = 0.5 * (prob_p + prob_q)
    log_mean = torch.log(mean_prob.clamp_min(1e-12))
    log_p = torch.log(prob_p.clamp_min(1e-12))
    log_q = torch.log(prob_q.clamp_min(1e-12))
    kl_pm = (prob_p * (log_p - log_mean)).sum()
    kl_qm = (prob_q * (log_q - log_mean)).sum()
    return float((0.5 * (kl_pm + kl_qm)).item())


def expand_routes(df: pd.DataFrame, sample_id_col: str, route_positions_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        sample_id = row[sample_id_col]
        positions = ensure_list_of_ints(row[route_positions_col]) or []
        for pos in positions:
            rows.append({sample_id_col: sample_id, "position": int(pos)})
    return pd.DataFrame(rows)


def get_top1_from_logits(logits: list[float]) -> int:
    return int(torch.tensor(logits, dtype=torch.float32).argmax().item())


def get_top1_from_topk(indices: list[int], logits: list[float]) -> int:
    if not indices or not logits:
        raise ValueError("Empty sparse top-k inputs.")
    max_idx = max(range(len(logits)), key=lambda i: logits[i])
    return int(indices[max_idx])


def summarize_series(series: pd.Series) -> dict[str, Optional[float]]:
    if len(series) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
        }
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.9)),
        "p95": float(series.quantile(0.95)),
    }


def maybe_fill_metrics(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.copy()

    has_full_logits = args.slm_logits_col in df.columns and args.llm_logits_col in df.columns
    has_sparse_topk = all(
        col in df.columns
        for col in (
            args.slm_topk_indices_col,
            args.slm_topk_logits_col,
            args.llm_topk_indices_col,
            args.llm_topk_logits_col,
        )
    )

    if args.slm_top1_col not in df.columns:
        if has_full_logits:
            df[args.slm_top1_col] = df[args.slm_logits_col].apply(
                lambda x: get_top1_from_logits(ensure_list_of_numbers(x))
            )
        elif has_sparse_topk:
            df[args.slm_top1_col] = df.apply(
                lambda r: get_top1_from_topk(
                    ensure_list_of_ints(r[args.slm_topk_indices_col]),
                    ensure_list_of_numbers(r[args.slm_topk_logits_col]),
                ),
                axis=1,
            )

    if args.llm_top1_col not in df.columns:
        if has_full_logits:
            df[args.llm_top1_col] = df[args.llm_logits_col].apply(
                lambda x: get_top1_from_logits(ensure_list_of_numbers(x))
            )
        elif has_sparse_topk:
            df[args.llm_top1_col] = df.apply(
                lambda r: get_top1_from_topk(
                    ensure_list_of_ints(r[args.llm_topk_indices_col]),
                    ensure_list_of_numbers(r[args.llm_topk_logits_col]),
                ),
                axis=1,
            )

    if args.slm_entropy_col not in df.columns and has_full_logits:
        df[args.slm_entropy_col] = df[args.slm_logits_col].apply(
            lambda x: compute_entropy_from_logits(ensure_list_of_numbers(x))
        )

    if args.js_col not in df.columns:
        if has_full_logits:
            df[args.js_col] = df.apply(
                lambda r: compute_js_from_logits(
                    ensure_list_of_numbers(r[args.slm_logits_col]),
                    ensure_list_of_numbers(r[args.llm_logits_col]),
                ),
                axis=1,
            )
        elif has_sparse_topk:
            df[args.js_col] = df.apply(
                lambda r: compute_sparse_topk_js(
                    ensure_list_of_ints(r[args.slm_topk_indices_col]),
                    ensure_list_of_numbers(r[args.slm_topk_logits_col]),
                    ensure_list_of_ints(r[args.llm_topk_indices_col]),
                    ensure_list_of_numbers(r[args.llm_topk_logits_col]),
                ),
                axis=1,
            )

    required = [args.slm_top1_col, args.llm_top1_col, args.slm_entropy_col, args.js_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Comparison file is missing required metrics and they could not be derived: "
            + ", ".join(missing)
        )

    return df


def main() -> None:
    args = parse_args()

    routes_df = load_table(args.routes)
    comparison_df = load_table(args.comparison)

    if args.sample_id_col not in routes_df.columns:
        raise ValueError(f"Routes file missing sample id column `{args.sample_id_col}`")
    if args.route_positions_col not in routes_df.columns:
        raise ValueError(f"Routes file missing route positions column `{args.route_positions_col}`")
    if args.sample_id_col not in comparison_df.columns:
        raise ValueError(f"Comparison file missing sample id column `{args.sample_id_col}`")
    if args.position_col not in comparison_df.columns:
        raise ValueError(f"Comparison file missing position column `{args.position_col}`")

    routed_positions_df = expand_routes(routes_df, args.sample_id_col, args.route_positions_col)
    if routed_positions_df.empty:
        raise ValueError("No routed positions found in routes file.")

    comparison_df = comparison_df.copy()
    comparison_df["position"] = comparison_df[args.position_col].astype(int)
    enriched_df = maybe_fill_metrics(comparison_df, args)

    merged = routed_positions_df.merge(
        enriched_df,
        on=[args.sample_id_col, "position"],
        how="left",
        validate="one_to_one",
    )
    merged["top1_same"] = merged[args.slm_top1_col] == merged[args.llm_top1_col]

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(f"{output_prefix}.csv", index=False)

    routed_total = int(len(merged))
    matched = int(merged["top1_same"].fillna(False).sum())
    different_df = merged[merged["top1_same"] == False].copy()  # noqa: E712
    unresolved = int(merged[[args.slm_top1_col, args.llm_top1_col]].isna().any(axis=1).sum())

    summary = {
        "routed_total": routed_total,
        "top1_same_count": matched,
        "top1_diff_count": int(len(different_df)),
        "top1_same_ratio": (matched / routed_total) if routed_total else None,
        "top1_diff_ratio": (len(different_df) / routed_total) if routed_total else None,
        "missing_metric_rows": unresolved,
        "top1_diff_slm_entropy": summarize_series(different_df[args.slm_entropy_col].dropna()),
        "top1_diff_js": summarize_series(different_df[args.js_col].dropna()),
        "top1_same_slm_entropy": summarize_series(
            merged[merged["top1_same"] == True][args.slm_entropy_col].dropna()  # noqa: E712
        ),
        "top1_same_js": summarize_series(
            merged[merged["top1_same"] == True][args.js_col].dropna()  # noqa: E712
        ),
    }

    with open(f"{output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote routed comparison rows to {output_prefix}.csv")
    print(f"Wrote summary to {output_prefix}_summary.json")


if __name__ == "__main__":
    main()

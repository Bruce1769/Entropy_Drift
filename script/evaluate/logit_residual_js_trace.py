#!/usr/bin/env python3
"""
From token_traces CSV rows that store quick_* and reference_* top-k logits, compute:

  JS( softmax(z_L - z_S), softmax(z_L) )

where z_L / z_S are aligned on a shared index set (see --mode).

Modes:
  union_imputed: union of quick/ref top-k indices; missing SLM (resp. LLM) logits imputed
         as min(quick_logits)-margin (resp. min(ref_logits)-margin) so z_L - z_S is finite.
         (Pure -inf fill makes z_L-z_S have many +inf and softmax becomes NaN.)
  intersection: only token ids present in BOTH top-k lists (requires |I| >= 2). Most faithful
         when only top-k logits are logged.

Requires torch (same JS as compute_js_divergence).
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# repo root on path
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _parse_list(cell: str) -> list:
    if not cell or not str(cell).strip():
        return []
    s = str(cell).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def js_divergence_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Same math as r2r.utils.metrics.compute_js_divergence (1-D)."""
    logits_p = logits_p.to(dtype=torch.float32).unsqueeze(0)
    logits_q = logits_q.to(dtype=torch.float32).unsqueeze(0)
    log_probs_p = F.log_softmax(logits_p, dim=-1)
    log_probs_q = F.log_softmax(logits_q, dim=-1)
    probs_p = log_probs_p.exp()
    probs_q = log_probs_q.exp()
    mean_probs = 0.5 * (probs_p + probs_q)
    log_mean_probs = torch.log(mean_probs.clamp_min(1e-12))
    kl_pm = torch.sum(probs_p * (log_probs_p - log_mean_probs), dim=-1)
    kl_qm = torch.sum(probs_q * (log_probs_q - log_mean_probs), dim=-1)
    return float((0.5 * (kl_pm + kl_qm))[0].item())


def js_residual_union_imputed(
    quick_ids: list,
    quick_logits: list,
    ref_ids: list,
    ref_logits: list,
    margin: float = 40.0,
) -> float | None:
    """Union support with finite imputation for missing top-k logits, then z_delta = z_L - z_S."""
    if not quick_ids or not ref_ids:
        return None
    q_i = torch.tensor(quick_ids, dtype=torch.long)
    q_lv = torch.tensor(quick_logits, dtype=torch.float32)
    r_i = torch.tensor(ref_ids, dtype=torch.long)
    r_lv = torch.tensor(ref_logits, dtype=torch.float32)

    combined = torch.cat([q_i, r_i], dim=0)
    union_indices, inverse = torch.unique(combined, sorted=True, return_inverse=True)
    q_positions = inverse[: q_i.shape[0]]
    r_positions = inverse[q_i.shape[0] :]

    q_floor = float(torch.min(q_lv).item()) - margin
    r_floor = float(torch.min(r_lv).item()) - margin

    z_s = torch.full((union_indices.shape[0],), q_floor, dtype=torch.float32)
    z_l = torch.full((union_indices.shape[0],), r_floor, dtype=torch.float32)
    z_s[q_positions] = q_lv
    z_l[r_positions] = r_lv

    z_delta = z_l - z_s
    return js_divergence_from_logits(z_delta, z_l)


def js_residual_intersection(
    quick_ids: list, quick_logits: list, ref_ids: list, ref_logits: list
) -> float | None:
    """Restrict to token ids in both top-k lists (first occurrence wins)."""
    dq = {int(i): float(l) for i, l in zip(quick_ids, quick_logits)}
    dr = {int(i): float(l) for i, l in zip(ref_ids, ref_logits)}
    inter = sorted(set(dq.keys()) & set(dr.keys()))
    if len(inter) < 2:
        return None
    z_s = torch.tensor([dq[i] for i in inter], dtype=torch.float32)
    z_l = torch.tensor([dr[i] for i in inter], dtype=torch.float32)
    z_delta = z_l - z_s
    return js_divergence_from_logits(z_delta, z_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "trace_glob",
        type=str,
        help="Glob for token_traces/*_run_1.csv (quoted if needed)",
    )
    ap.add_argument(
        "--mode",
        choices=("union_imputed", "intersection"),
        default="intersection",
        help="How to align logits (default: intersection; union uses floor imputation for gaps)",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=40.0,
        help="Imputation gap below min observed logit per side (union_imputed only)",
    )
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.trace_glob))
    if not paths:
        print("No files matched:", args.trace_glob, file=sys.stderr)
        sys.exit(1)

    # (not `fn = lambda ...: a if cond else b` — that binds the ternary inside the lambda body)
    if args.mode == "union_imputed":

        def fn(qid, ql, rid, rl):
            return js_residual_union_imputed(qid, ql, rid, rl, margin=args.margin)

    else:
        fn = js_residual_intersection
    assert args.mode in ("union_imputed", "intersection")
    vals: list[float] = []
    skipped = 0
    rows_in = 0

    for path in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_in += 1
                if args.max_rows and rows_in > args.max_rows:
                    break
                qids = _parse_list(row.get("quick_topk_token_ids") or "")
                ql = _parse_list(row.get("quick_topk_logits") or "")
                rids = _parse_list(row.get("reference_topk_token_ids") or "")
                rl = _parse_list(row.get("reference_topk_logits") or "")
                if len(qids) != len(ql) or len(rids) != len(rl):
                    skipped += 1
                    continue
                if not qids or not rids:
                    skipped += 1
                    continue
                j = fn(qids, ql, rids, rl)
                if j is None:
                    skipped += 1
                    continue
                try:
                    jf = float(j)
                except (TypeError, ValueError):
                    skipped += 1
                    continue
                if not math.isfinite(jf):
                    skipped += 1
                    continue
                vals.append(jf)
        if args.max_rows and rows_in > args.max_rows:
            break

    arr = np.array(vals, dtype=np.float64)
    print(f"files: {len(paths)}  rows_read: {rows_in}  computed: {len(vals)}  skipped: {skipped}")
    print(f"mode: {args.mode}")
    if len(arr) == 0:
        print("No JS values computed (missing top-k columns or empty intersection).")
        sys.exit(2)
    for name, q in [
        ("mean", np.mean),
        ("std", np.std),
        ("min", np.min),
        ("p25", lambda a: np.quantile(a, 0.25)),
        ("p50", lambda a: np.quantile(a, 0.50)),
        ("p75", lambda a: np.quantile(a, 0.75)),
        ("max", np.max),
    ]:
        print(f"{name}: {float(q(arr)):.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Grid-search a *train-free* combined score from SLM entropy H and a top-1 vs top-2 margin,
to approximate NeuralSwitching's router_decision on saved token_traces.

Uses per-column z-scores (global across all loaded rows) and linear decision:
    pred = 1[ wH * z(H) + wM * z(fM) > tau ]

Margin fM (higher => more "risk" / more like escalating):
  - If quick_topk_logits has >= 2 values: fM = -(z1 - z2)  so larger when top1/top2 tight
    (we negate so that it aligns with entropy direction; equivalently use + (z2-z1)).
  - Else: fM = 1 - p1  (mass not on argmax; no p2 available — documented proxy)

Weights (wH, wM) are constrained to the unit circle (wH^2+wM^2=1), tau scanned on quantiles
of the combined score distribution.

Reports best F1 vs neural, accuracy, and predicted switch rate vs neural rate.

Example:
  python3 combined_entropy_margin_router_grid.py \\
    "/remote-home/pxl/R2R/output/eval/.../token_traces/*_run_1.csv"
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


def f1_binary_np(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    if tp == 0:
        return 0.0
    return float(2 * tp / (2 * tp + fp + fn))


def acc_binary_np(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(y == pred))


def _parse_list(cell: str) -> list:
    if not cell or not str(cell).strip():
        return []
    s = str(cell).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def load_arrays(paths: list[str]):
    y: list[int] = []
    h: list[float] = []
    fM: list[float] = []
    used_true_margin: list[bool] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                try:
                    yi = int(row["router_decision"])
                    hi = float(row["quick_entropy"])
                    p1 = float(row["quick_top1_prob"])
                except (KeyError, ValueError):
                    continue
                ql = _parse_list(row.get("quick_topk_logits") or "")
                if len(ql) >= 2:
                    z1, z2 = float(ql[0]), float(ql[1])
                    # risk: small logit gap => escalate; use negative gap so z-scores align with H
                    mi = -(z1 - z2)
                    used_true_margin.append(True)
                else:
                    mi = 1.0 - p1
                    used_true_margin.append(False)
                y.append(yi)
                h.append(hi)
                fM.append(mi)
    return (
        np.asarray(y, dtype=np.int64),
        np.asarray(h, dtype=np.float64),
        np.asarray(fM, dtype=np.float64),
        used_true_margin,
    )


def zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean()
    s = x.std()
    if s <= 1e-12:
        return np.zeros_like(x)
    return (x - m) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_glob", type=str)
    ap.add_argument(
        "--angle-steps",
        type=int,
        default=181,
        help="Number of angles in [0, pi/2] for (wH, wM)",
    )
    ap.add_argument(
        "--tau-quantiles",
        type=int,
        default=400,
        help="Number of tau thresholds from score quantiles",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(args.trace_glob))
    if not paths:
        print("No files:", args.trace_glob, file=sys.stderr)
        sys.exit(1)

    y, h, fM, used_true = load_arrays(paths)
    n = len(y)
    if n < 100:
        print("Too few rows:", n, file=sys.stderr)
        sys.exit(2)

    true_margin_rate = float(np.mean(used_true))
    print(
        f"rows={n}  neural_switch_rate={y.mean():.6f}  "
        f"fraction_rows_with_true_logit_margin={true_margin_rate:.4f}"
    )

    zH = zscore(h)
    zM = zscore(fM)

    neural_rate = float(y.mean())
    # best: (f1, neg_rate_err, acc, pr, wH, wM, tau)
    best = (-1.0, -1e9, 0.0, 0.0, 0.0, 0.0, 0.0)

    def better(candidate, current):
        f1_c, re_c = candidate
        f1_o, re_o = current[0], current[1]
        if f1_c > f1_o + 1e-9:
            return True
        if math.isclose(f1_c, f1_o, rel_tol=1e-9) and re_c > re_o + 1e-12:
            return True
        return False

    for k in range(args.angle_steps):
        theta = (math.pi / 2) * (k / max(args.angle_steps - 1, 1))
        wH = math.cos(theta)
        wM = math.sin(theta)
        s = wH * zH + wM * zM
        taus = np.unique(np.quantile(s, np.linspace(0, 1, args.tau_quantiles)))
        for tau in taus:
            pred = (s > tau).astype(np.int64)
            acc = acc_binary_np(y, pred)
            f1 = f1_binary_np(y, pred)
            pr = float(pred.mean())
            rate_err = -abs(pr - neural_rate)  # larger is better (closer rates)
            cand = (f1, rate_err)
            cur = (best[0], best[1])
            if better(cand, cur):
                best = (f1, rate_err, acc, pr, wH, wM, tau)

    f1_b, _, acc_b, pr_b, wH_b, wM_b, tau_b = best

    # Baseline: entropy-only on zH (max F1)
    sH = zH
    best_h = (-1.0, 0.0, 0.0)  # f1, pred_rate, tau
    for tau in np.unique(np.quantile(sH, np.linspace(0, 1, args.tau_quantiles))):
        pred = (sH > tau).astype(np.int64)
        f1 = f1_binary_np(y, pred)
        pr = float(pred.mean())
        if f1 > best_h[0] + 1e-12:
            best_h = (f1, pr, tau)

    print("\n--- Baseline: best single-feature z(entropy) threshold (max F1) ---")
    print(f"best F1={best_h[0]:.4f}  pred_switch_rate={best_h[1]:.6f}  (neural={neural_rate:.6f})")

    print("\n--- Best linear combo: pred = 1[ wH*z(H) + wM*z(fM) > tau ] ---")
    print(f"wH={wH_b:.4f}  wM={wM_b:.4f}  (theta={math.degrees(math.atan2(wM_b, wH_b)):.2f} deg)")
    print(f"tau={tau_b:.6f}")
    print(f"F1={f1_b:.4f}  acc={acc_b:.4f}  pred_switch_rate={pr_b:.6f}  neural_switch_rate={neural_rate:.6f}")
    print(
        "\nNote: if quick_topk_logits is missing, fM = 1 - p1 (proxy). "
        "Re-run eval with --trace_logits_topk_k >= 2 for true logit margin."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze relationship between SLM quick_entropy and SLM vs LLM top-5 overlap / rank shift.

Expects token_traces CSV with quick_topk_token_ids, reference_topk_token_ids
(logits order = descending score from torch.topk in the server).

Metrics per row (when both sides have >=5 ids):
  - overlap_5: |set(slm_top5) ∩ set(llm_top5)|
  - mean_abs_rank_diff: mean over tokens in intersection of |rank_slm(t) - rank_llm(t)|
  - top1_same: slm_top5[0] == llm_top5[0]

Then: Pearson/Spearman between quick_entropy and overlap_5 (and other metrics);
entropy deciles with mean overlap.
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


def _parse_list(cell: str) -> list:
    if not cell or not str(cell).strip():
        return []
    s = str(cell).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def _ranks_top5(ids: list[int]) -> dict[int, int]:
    """rank 0 = highest logit (top-1)."""
    out: dict[int, int] = {}
    for r, tid in enumerate(ids[:5]):
        out[int(tid)] = r
    return out


def row_metrics(qids: list, rids: list) -> tuple[int | None, float | None, int | None]:
    if len(qids) < 5 or len(rids) < 5:
        return None, None, None
    s5 = [int(x) for x in qids[:5]]
    l5 = [int(x) for x in rids[:5]]
    ss, ls = set(s5), set(l5)
    overlap = len(ss & ls)
    rs = _ranks_top5(s5)
    rl = _ranks_top5(l5)
    inter = ss & ls
    if not inter:
        mad = float("nan")
    else:
        mad = sum(abs(rs[t] - rl[t]) for t in inter) / len(inter)
    top1_same = 1 if s5[0] == l5[0] else 0
    return overlap, mad, top1_same


def _corr_pearson(x: list[float], y: list[float]) -> tuple[float, int]:
    n = len(x)
    if n < 3:
        return float("nan"), n
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return float("nan"), n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    return cov / math.sqrt(vx * vy), n


def _corr_spearman(x: list[float], y: list[float]) -> tuple[float, int]:
    """Spearman via rank transform (midrank for ties — here we use dense ranks)."""
    n = len(x)
    if n < 3:
        return float("nan"), n

    def ranks(vals: list[float]) -> list[float]:
        idx = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[idx[j + 1]] == vals[idx[i]]:
                j += 1
            mid = (i + j + 1) / 2.0
            for k in range(i, j + 1):
                r[idx[k]] = mid
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    return _corr_pearson(rx, ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_glob", type=str)
    ap.add_argument(
        "--entropy-column",
        default="quick_entropy",
        help="SLM entropy column (default quick_entropy)",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(args.trace_glob))
    if not paths:
        print("No files:", args.trace_glob, file=sys.stderr)
        sys.exit(1)

    h_list: list[float] = []
    ov_list: list[float] = []
    mad_list: list[float] = []
    t1_list: list[float] = []

    skipped = 0
    rows = 0
    for path in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                rows += 1
                qids = _parse_list(row.get("quick_topk_token_ids") or "")
                rids = _parse_list(row.get("reference_topk_token_ids") or "")
                try:
                    h = float(row[args.entropy_column])
                except (KeyError, TypeError, ValueError):
                    skipped += 1
                    continue
                ov, mad, t1 = row_metrics(qids, rids)
                if ov is None:
                    skipped += 1
                    continue
                h_list.append(h)
                ov_list.append(float(ov))
                if mad is not None and not math.isnan(mad):
                    mad_list.append((h, mad))
                t1_list.append((h, float(t1)))

    n = len(h_list)
    print(f"files {len(paths)}  rows_read {rows}  usable {n}  skipped {skipped}")
    if n < 10:
        print("Too few rows.", file=sys.stderr)
        sys.exit(2)

    rp, _ = _corr_pearson(h_list, ov_list)
    rs, _ = _corr_spearman(h_list, ov_list)
    print(f"Pearson( entropy, overlap_5 ) = {rp:.4f}")
    print(f"Spearman( entropy, overlap_5 ) = {rs:.4f}")

    if mad_list:
        mh = [a for a, _ in mad_list]
        mm = [b for _, b in mad_list]
        print(
            f"Pearson( entropy, mean_abs_rank_diff_on_intersection ) = {_corr_pearson(mh, mm)[0]:.4f}  (n={len(mad_list)})"
        )
    ht1 = [a for a, _ in t1_list]
    yt1 = [b for _, b in t1_list]
    print(f"Pearson( entropy, top1_same ) = {_corr_pearson(ht1, yt1)[0]:.4f}")

    # Deciles by rank (equal-count-ish): sort by entropy, assign decile by rank
    order = sorted(range(n), key=lambda i: h_list[i])
    dec_of = [0] * n
    for rank, idx in enumerate(order):
        dec_of[idx] = min(9, int(rank * 10 / max(n, 1)))

    bins = [[] for _ in range(10)]
    for i in range(n):
        bins[dec_of[i]].append(ov_list[i])

    print("\nMean overlap_5 by entropy decile (0=lowest entropy, 9=highest):")
    for d in range(10):
        b = bins[d]
        if not b:
            print(f"  d{d}: n=0")
        else:
            print(f"  d{d}: n={len(b):6d}  mean_overlap={sum(b)/len(b):.4f}")

    # Optional: mean entropy per overlap bucket
    print("\nMean entropy given overlap_5:")
    for k in range(6):
        sub = [h for h, o in zip(h_list, ov_list) if int(o) == k]
        if sub:
            print(f"  overlap={k}: n={len(sub):6d}  mean_entropy={sum(sub)/len(sub):.4f}")


if __name__ == "__main__":
    main()

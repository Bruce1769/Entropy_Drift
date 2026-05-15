#!/usr/bin/env python3
"""
Ablation-style report: how JS classification labels change with threshold on js_value,
and optional agreement with on-disk js_high (from build_js_entropy_risk_dataset).

Usage:
  PYTHONPATH=/root/Entropy_Drift /root/miniconda3/envs/r2r/bin/python \\
    script/train/report_js_label_sensitivity.py \\
    --val-path /root/autodl-tmp/datasets/js_entropy_risk_dataset/validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ED = Path(__file__).resolve().parents[2]
if str(_ED) not in sys.path:
    sys.path.insert(0, str(_ED))

from datasets import load_from_disk  # noqa: E402

from r2r.utils.router_training_proxy import js_high_from_js_values  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--val-path",
        type=str,
        default="/root/autodl-tmp/datasets/js_entropy_risk_dataset/validation",
    )
    ap.add_argument("--max-rows", type=int, default=500_000)
    ap.add_argument(
        "--tau-grid",
        type=str,
        default="0.05,0.08,0.1,0.12,0.15",
        help="Comma-separated JS thresholds for recomputed js_high",
    )
    args = ap.parse_args()

    ds = load_from_disk(args.val_path)
    n = min(len(ds), args.max_rows)
    js = np.asarray(ds["js_value"][:n], dtype=np.float32)
    disk = np.asarray(ds["js_high"][:n], dtype=np.int64)
    ent = np.asarray(ds["entropy_value"][:n], dtype=np.float32)

    taus = [float(x.strip()) for x in args.tau_grid.split(",") if x.strip()]
    print(f"Rows used: {n}")
    print("tau_js\tpos_rate\tagreement_with_disk_js_high")
    for tau in taus:
        y = js_high_from_js_values(js, tau)
        agree = float((y == disk).mean())
        pr = float(y.mean())
        print(f"{tau}\t{pr:.6f}\t{agree:.6f}")

    # Entropy deciles vs mean JS (sanity)
    qs = np.quantile(ent, np.linspace(0, 1, 11))
    print("\nentropy decile edges:", np.array2string(qs, precision=4))


if __name__ == "__main__":
    main()

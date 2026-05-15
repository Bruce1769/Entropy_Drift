#!/usr/bin/env python3
"""
Offline grid search for probability threshold on a trained RouterMultiTaskFFN4 checkpoint,
using the same validation tensors as train_router_multitask_js.py (no SGLang).

Optionally filter rows by entropy < entropy_subset_thr to mimic inference under entropy gate.

Usage:
  PYTHONPATH=/root/Entropy_Drift /root/miniconda3/envs/r2r/bin/python \\
    script/train/sweep_router_prob_threshold.py \\
    --ckpt /root/autodl-tmp/datasets/router_multitask_js_runs/run_v2_recall/best.pt \\
    --val-path /root/autodl-tmp/datasets/js_entropy_risk_dataset/validation \\
    --router-repr-dir /root/autodl-tmp/datasets/prefill_1p5b_vs_32b_logits/router_repr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score

_ED = Path(__file__).resolve().parents[2]
if str(_ED) not in sys.path:
    sys.path.insert(0, str(_ED))

from datasets import load_from_disk  # noqa: E402

from r2r.models.multitask_js_router import RouterMultiTaskFFN4  # noqa: E402
from r2r.utils.router_training_proxy import js_high_from_js_values  # noqa: E402


def load_logits_js_ent_labels(path: Path, max_rows: int | None):
    ds = load_from_disk(str(path))
    n = len(ds) if max_rows is None else min(len(ds), max_rows)
    sl = np.asarray(ds["small_logits"][:n], dtype=np.float32)
    logits100 = sl[:, :100].copy()
    js = np.asarray(ds["js_value"][:n], dtype=np.float32)
    ent = np.asarray(ds["entropy_value"][:n], dtype=np.float32)
    run_index = np.asarray(ds["run_index"][:n], dtype=np.int64)
    token_pos = np.asarray(ds["token_pos"][:n], dtype=np.int32)
    return logits100, js, ent, run_index, token_pos


def materialize_rows(run_index: np.ndarray, token_pos: np.ndarray, repr_dir: Path, hidden_dim: int):
    n = len(run_index)
    order = np.argsort(run_index, kind="mergesort")
    ri_sorted = run_index[order]
    pos_sorted = token_pos[order]
    hid = np.zeros((n, hidden_dim), dtype=np.float32)
    tok = np.zeros(n, dtype=np.int64)
    i = 0
    while i < n:
        rid = int(ri_sorted[i])
        j = i
        while j < n and int(ri_sorted[j]) == rid:
            j += 1
        z = np.load(repr_dir / f"run_{rid:04d}.npz", mmap_mode="r")
        H = z["last_hidden_state"]
        T = z["routing_input_token_ids"]
        block_idx = order[i:j]
        local_pos = pos_sorted[i:j].astype(np.int64)
        hid[block_idx] = np.asarray(H[local_pos], dtype=np.float32)
        tok[block_idx] = np.asarray(T[local_pos], dtype=np.int64)
        i = j
    return hid, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument(
        "--val-path",
        type=str,
        default="/root/autodl-tmp/datasets/js_entropy_risk_dataset/validation",
    )
    ap.add_argument(
        "--router-repr-dir",
        type=str,
        default="/root/autodl-tmp/datasets/prefill_1p5b_vs_32b_logits/router_repr",
    )
    ap.add_argument("--js-cls-threshold", type=float, default=0.1)
    ap.add_argument("--entropy-subset-thr", type=float, default=1.0)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--prob-min", type=float, default=0.2)
    ap.add_argument("--prob-max", type=float, default=0.55)
    ap.add_argument("--prob-steps", type=int, default=15)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits100, js, ent, run_index, token_pos = load_logits_js_ent_labels(Path(args.val_path), args.max_rows)
    y = js_high_from_js_values(js, args.js_cls_threshold)

    repr_dir = Path(args.router_repr_dir)
    hid, tok = materialize_rows(run_index, token_pos, repr_dir, hidden_dim=1536)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved = ckpt.get("args") or {}
    model = RouterMultiTaskFFN4(
        ffn_dim=int(saved.get("ffn_dim", 768)),
        dropout=float(saved.get("dropout", 0.12)),
        normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
        freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
        pretrained_model_name=saved.get("pretrained_model_name") or "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    lo = torch.from_numpy(logits100).to(device)
    hh = torch.from_numpy(hid).to(device)
    tt = torch.from_numpy(tok).to(device)
    with torch.no_grad():
        cls_logits, _ = model(lo, hh, tt)
        probs = torch.sigmoid(cls_logits.squeeze(-1)).cpu().numpy()

    low_ent = ent < args.entropy_subset_thr
    print(f"rows={len(y)} low_entropy_frac={low_ent.mean():.4f} (thr={args.entropy_subset_thr})")
    print("prob_thr\tf1_all\trec_all\tf1_low_ent\trec_low_ent\tpos_rate_low_ent")

    for t in np.linspace(args.prob_min, args.prob_max, args.prob_steps):
        pred = (probs >= t).astype(np.int32)
        f1 = f1_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, pos_label=1, zero_division=0)
        if np.any(low_ent):
            ye, pe = y[low_ent], pred[low_ent]
            f1e = f1_score(ye, pe, zero_division=0)
            rece = recall_score(ye, pe, pos_label=1, zero_division=0)
            pr = float(pe.mean())
        else:
            f1e = rece = pr = float("nan")
        print(f"{t:.4f}\t{f1:.4f}\t{rec:.4f}\t{f1e:.4f}\t{rece:.4f}\t{pr:.4f}")

    if "best_prob_threshold" in ckpt:
        print(f"\nckpt best_prob_threshold: {ckpt['best_prob_threshold']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Offline calibrate entropy_drift knobs against R2R neural token_traces.

Reads token_traces/*.csv from an eval run (NeuralSwitching): uses quick_entropy as h_t,
quick_top1_prob for max_confident_prob gating, and router_decision as the label (0=quick, 1=reference).

Simulates EntropyDriftSwitching with stochastic=False to match r2r/utils/switching.py.

Example:
  python script/evaluate/calibrate_entropy_drift_to_r2r_traces.py \\
    --trace-dir output/eval/aime26_r2r_neural_t1_idle23_20260416_035759/token_traces \\
    --alpha-values 0.05,0.1,0.15,0.2,0.3 \\
    --bias-values 0.15,0.2,0.25,0.3,0.35 \\
    --hold-out-problem-ids 25,26,27,28,29,30
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SimConfig:
    alpha: float
    bias: float
    warmup_steps: int
    hysteresis: float
    hold_tokens: int
    max_confident_prob: Optional[float]
    min_entropy: Optional[float]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _float_range(lo: float, hi: float, step: float) -> List[float]:
    """Inclusive of endpoints on the grid (within floating tolerance)."""
    if step <= 0:
        raise ValueError("step must be positive")
    out: List[float] = []
    x = lo
    while x <= hi + 1e-9:
        out.append(round(x, 10))
        x += step
    return out


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_optional_int_set(s: Optional[str]) -> Optional[set]:
    if s is None or not str(s).strip():
        return None
    return set(_parse_int_list(s))


def _problem_id_from_filename(path: str) -> int:
    base = os.path.basename(path)
    m = re.match(r"^(\d+)_run_1\.csv$", base)
    if not m:
        raise ValueError(f"Unexpected trace filename: {base}")
    return int(m.group(1))


def load_trace_rows(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (entropy, top1_prob, router_decision) float arrays, same length."""
    h_list: List[float] = []
    t1_list: List[float] = []
    y_list: List[int] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                h = float(row["quick_entropy"])
            except (KeyError, ValueError, TypeError):
                continue
            raw_t1 = row.get("quick_top1_prob", "")
            try:
                t1 = float(raw_t1) if str(raw_t1).strip() != "" else float("nan")
            except (ValueError, TypeError):
                t1 = float("nan")
            try:
                y = int(float(row["router_decision"]))
            except (KeyError, ValueError, TypeError):
                y = 0
            h_list.append(h)
            t1_list.append(t1)
            y_list.append(y)
    return (
        np.asarray(h_list, dtype=np.float64),
        np.asarray(t1_list, dtype=np.float64),
        np.asarray(y_list, dtype=np.int8),
    )


def passes_secondary_filter(
    h_t: float,
    top1_prob: float,
    min_entropy: Optional[float],
    max_confident_prob: Optional[float],
) -> bool:
    if min_entropy is not None and h_t < min_entropy:
        return False
    if max_confident_prob is not None:
        if not math.isfinite(top1_prob):
            # Missing top1: do not apply the confidence gate (treat as unknown).
            return True
        if top1_prob > max_confident_prob:
            return False
    return True


def simulate_entropy_drift_deterministic(
    h: np.ndarray,
    top1: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """
    Token-level predictions (0/1) matching EntropyDriftSwitching.route, stochastic=False.
    """
    n = len(h)
    pred = np.zeros(n, dtype=np.int8)
    ema_mean = 0.0
    n_seen = 0
    last_model = "quick"
    hold_remaining = 0

    for t in range(n):
        h_t = float(h[t])
        t1 = float(top1[t]) if t < len(top1) else float("nan")

        if n_seen < cfg.warmup_steps:
            pred[t] = 0
        else:
            drift = h_t - ema_mean
            pf = passes_secondary_filter(
                h_t, t1, cfg.min_entropy, cfg.max_confident_prob
            )
            enter_th = cfg.bias
            exit_th = cfg.bias - cfg.hysteresis
            if last_model == "reference":
                choose_ref = pf and (hold_remaining > 0 or drift >= exit_th)
            else:
                choose_ref = pf and (drift >= enter_th)
            pred[t] = 1 if choose_ref else 0

        if int(pred[t]) == 1:
            if last_model != "reference":
                hold_remaining = cfg.hold_tokens
            elif hold_remaining > 0:
                hold_remaining -= 1
            last_model = "reference"
        else:
            hold_remaining = 0
            last_model = "quick"

        if n_seen == 0:
            ema_mean = h_t
        else:
            ema_mean = cfg.alpha * h_t + (1.0 - cfg.alpha) * ema_mean
        n_seen += 1

    return pred


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred in {0,1}. Reports accuracy, precision/recall/F1 for class 1."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {
        "accuracy": acc,
        "precision_ref": prec,
        "recall_ref": rec,
        "f1_ref": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(y_true),
    }


def iter_trace_files(trace_dir: str) -> List[str]:
    paths = sorted(
        glob.glob(os.path.join(trace_dir, "*_run_1.csv")),
        key=_problem_id_from_filename,
    )
    if not paths:
        raise FileNotFoundError(f"No *_run_1.csv under {trace_dir}")
    return paths


def aggregate_metrics_for_sequences(
    sequences: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    cfg: SimConfig,
    skip_warmup_tokens: bool,
    warmup_steps: int,
) -> dict:
    """sequences: list of (h, top1, y) arrays per problem."""
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    for h, t1, y in sequences:
        pred = simulate_entropy_drift_deterministic(h, t1, cfg)
        if skip_warmup_tokens and len(y) > warmup_steps:
            pred = pred[warmup_steps:]
            y = y[warmup_steps:]
        preds.append(pred)
        trues.append(y)
    if not preds:
        return {k: float("nan") for k in ["accuracy", "f1_ref", "precision_ref", "recall_ref"]}
    pcat = np.concatenate(preds)
    ycat = np.concatenate(trues)
    return binary_metrics(ycat, pcat)


def load_sequences_for_paths(paths: Sequence[str]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [load_trace_rows(p) for p in paths]


def aggregate_metrics_for_paths(
    paths: Sequence[str],
    cfg: SimConfig,
    skip_warmup_tokens: bool,
    warmup_steps: int,
) -> dict:
    return aggregate_metrics_for_sequences(
        load_sequences_for_paths(paths), cfg, skip_warmup_tokens, warmup_steps
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search entropy_drift params to match R2R token_traces."
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        required=True,
        help="Directory containing token_traces/*_run_1.csv from a NeuralSwitching eval.",
    )
    parser.add_argument(
        "--hold-out-problem-ids",
        type=str,
        default="",
        help="Comma-separated problem ids for validation only (e.g. 25,26,...,30). Empty = no hold-out.",
    )
    parser.add_argument(
        "--alpha-values",
        type=str,
        default="0.05,0.08,0.1,0.15,0.2,0.25,0.3",
    )
    parser.add_argument(
        "--alpha-range",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        default=None,
        help="If set, overrides --alpha-values (e.g. 0.05 0.35 0.01).",
    )
    parser.add_argument(
        "--bias-values",
        type=str,
        default="0.1,0.15,0.2,0.25,0.3,0.35,0.4",
    )
    parser.add_argument(
        "--bias-range",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        default=None,
        help="If set, overrides --bias-values (e.g. 0.10 0.50 0.01).",
    )
    parser.add_argument(
        "--warmup-values",
        type=str,
        default="32",
    )
    parser.add_argument("--hysteresis-values", type=str, default="0.15")
    parser.add_argument("--hold-tokens-values", type=str, default="4")
    parser.add_argument("--max-confident-prob-values", type=str, default="0.85")
    parser.add_argument(
        "--min-entropy",
        type=float,
        default=None,
        help="If set, same as drift_min_entropy (often leave unset for calibration).",
    )
    parser.add_argument(
        "--skip-warmup-tokens-in-metrics",
        action="store_true",
        help="Exclude the first warmup_steps tokens when computing metrics (matches post-warmup only).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Print this many best configs by train F1.",
    )
    args = parser.parse_args()

    trace_dir = os.path.abspath(args.trace_dir)
    all_paths = iter_trace_files(trace_dir)
    hold_out = _parse_optional_int_set(args.hold_out_problem_ids)

    train_paths = [p for p in all_paths if hold_out is None or _problem_id_from_filename(p) not in hold_out]
    val_paths = [p for p in all_paths if hold_out is not None and _problem_id_from_filename(p) in hold_out]

    if hold_out is not None and not val_paths:
        raise SystemExit(
            f"No trace files found for hold-out ids {sorted(hold_out)} under {trace_dir}"
        )
    if not train_paths:
        raise SystemExit("No training trace files after hold-out split.")

    if args.alpha_range is not None:
        lo, hi, st = args.alpha_range
        alphas = _float_range(lo, hi, st)
    else:
        alphas = _parse_float_list(args.alpha_values)
    if args.bias_range is not None:
        lo, hi, st = args.bias_range
        biases = _float_range(lo, hi, st)
    else:
        biases = _parse_float_list(args.bias_values)
    warmups = _parse_int_list(args.warmup_values)
    hysts = _parse_float_list(args.hysteresis_values)
    holds = _parse_int_list(args.hold_tokens_values)
    mcp_raw = [x.strip() for x in args.max_confident_prob_values.split(",") if x.strip()]
    mcps: List[Optional[float]] = []
    for x in mcp_raw:
        if x.lower() in ("none", "nan", "null"):
            mcps.append(None)
        else:
            mcps.append(float(x))

    grid = list(
        itertools.product(alphas, biases, warmups, hysts, holds, mcps)
    )
    print(f"Trace dir: {trace_dir}")
    print(f"Train problems: {len(train_paths)}  Val problems: {len(val_paths)}")
    print(f"Grid size: {len(grid)} configs\n")

    train_seqs = load_sequences_for_paths(train_paths)
    val_seqs = load_sequences_for_paths(val_paths) if val_paths else []

    results: List[Tuple[float, SimConfig, dict, dict]] = []

    for alpha, bias, warmup, hysteresis, hold_tokens, mcp in grid:
        cfg = SimConfig(
            alpha=alpha,
            bias=bias,
            warmup_steps=warmup,
            hysteresis=hysteresis,
            hold_tokens=hold_tokens,
            max_confident_prob=mcp,
            min_entropy=args.min_entropy,
        )
        tr = aggregate_metrics_for_sequences(
            train_seqs,
            cfg,
            args.skip_warmup_tokens_in_metrics,
            warmup,
        )
        va = (
            aggregate_metrics_for_sequences(
                val_seqs,
                cfg,
                args.skip_warmup_tokens_in_metrics,
                warmup,
            )
            if val_seqs
            else {}
        )
        f1_tr = tr.get("f1_ref", float("nan"))
        results.append((f1_tr, cfg, tr, va))

    results.sort(
        key=lambda x: (
            -(x[0] if math.isfinite(x[0]) else -1.0),
            -x[2].get("accuracy", 0.0),
            -(x[3].get("f1_ref", 0.0) if x[3] else 0.0),
        )
    )

    print(f"Top {args.top_k} by train F1 (reference class):")
    print(
        "alpha  bias  warmup  hyst  hold  mcp      train_acc  train_F1  train_P  train_R  "
        + ("val_acc  val_F1  val_P  val_R" if val_paths else "")
    )
    for f1_tr, cfg, tr, va in results[: args.top_k]:
        mcp_s = "None" if cfg.max_confident_prob is None else f"{cfg.max_confident_prob:g}"
        line = (
            f"{cfg.alpha:5.3f}  {cfg.bias:5.3f}  {cfg.warmup_steps:6d}  {cfg.hysteresis:4.2f}  "
            f"{cfg.hold_tokens:4d}  {mcp_s:7s}  "
            f"{tr['accuracy']:9.5f}  {tr['f1_ref']:8.5f}  {tr['precision_ref']:7.5f}  {tr['recall_ref']:7.5f}"
        )
        if val_paths:
            line += (
                f"  {va['accuracy']:7.5f}  {va['f1_ref']:6.5f}  {va['precision_ref']:6.5f}  {va['recall_ref']:6.5f}"
            )
        print(line)

    best_f1, best_cfg, best_tr, best_va = results[0]
    print("\nBest config (by train F1):")
    print(
        f"  --drift_alpha {best_cfg.alpha} --drift_bias {best_cfg.bias} "
        f"--drift_warmup_steps {best_cfg.warmup_steps} --drift_hysteresis {best_cfg.hysteresis} "
        f"--drift_hold_tokens {best_cfg.hold_tokens}"
        + (
            f" --drift_max_confident_prob {best_cfg.max_confident_prob}"
            if best_cfg.max_confident_prob is not None
            else "  # omit --drift_max_confident_prob or set 1.0 to disable gate"
        )
    )
    print(f"  Train token accuracy={best_tr['accuracy']:.5f} F1_ref={best_tr['f1_ref']:.5f}")
    if val_paths:
        print(f"  Val   token accuracy={best_va['accuracy']:.5f} F1_ref={best_va['f1_ref']:.5f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train an R2R-style neural router (same inputs as HiddenStatesTokenLMHeadLogitsClassifier:
top-k logits + last-layer hidden + token id), with a 4-layer FFN trunk and multi-task loss.

Main task: binary classification for js_high (default: use on-disk label from dataset builder,
often js > tau_js in manifest; override with --js_cls_threshold to recompute from js_value, e.g. 0.1).

Auxiliary (optional): JS regression with (Focal-style regression term + MSE), weighted by --aux_weight (default 0.2).

Data: js_entropy_risk_dataset (HF on disk) + router_repr/*.npz for hidden states and token ids.
Stratified batches: positive : negative = 1 : 4; 60% of negatives are hard (0.05 < JS <= 0.1).

After training: PR curve on validation; recall on entropy < threshold at prob 0.5 and at
val-F1-optimal threshold. Model always uses token id -> nn.Embedding in the fusion path
(default: trainable; use --freeze_token_embeddings to freeze the 1.5B copy). LR: cosine decay
with linear warmup over warmup_epoch_frac. Early stopping when val PR-AUC and val F1 both
fail to improve (by min-deltas) for early_stop_patience epochs (default 64 max epochs, 8 patience).
Best checkpoint maximizes val_pr_auc + checkpoint_score_f1_weight * val_f1.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_ED = Path("/root/Entropy_Drift")
if str(_ED) not in sys.path:
    sys.path.insert(0, str(_ED))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import auc, f1_score, precision_recall_curve, recall_score
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm
from r2r.models.multitask_js_router import RouterMultiTaskFFN4, RouterBottleneck2Block, RouterBottleneck3Block, RouterBottleneck3BlockV7, RouterBottleneck3BlockV8, RouterBottleneckR2R
from r2r.utils.router_training_proxy import entropy_topk_shannon_logits, js_high_from_js_values
from r2r.utils.counterfactual_router_labels import load_counterfactual_map, merge_js_high_with_counterfactual

MODEL15_PATH = "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/datasets/js_entropy_risk_dataset")
DEFAULT_ROUTER_REPR = Path(
    "/root/autodl-tmp/datasets/prefill_1p5b_vs_32b_logits/router_repr"
)


def classification_target(js: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    return (js > threshold).float()


def apply_boundary_smoothing(
    js: torch.Tensor, y_hard: torch.Tensor, lo: float = 0.08, hi: float = 0.12
) -> torch.Tensor:
    """Linear soft targets only for js in [lo, hi]; outside band keep hard labels."""
    mask = (js >= lo) & (js <= hi)
    y_soft = ((js - lo) / (hi - lo)).clamp(0.0, 1.0)
    return torch.where(mask, y_soft, y_hard)


class FocalMSELoss(nn.Module):
    """Focal-style emphasis on larger errors: mean( (1 - exp(-se))^gamma * se )."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        se = (pred - target) ** 2
        pt = torch.exp(-se.clamp(max=20.0))
        focal = (1.0 - pt).pow(self.gamma) * se
        return focal.mean()


class StratifiedRouterBatchSampler(BatchSampler):
    """Each batch: n_pos : n_neg = 1 : 4, sampled from js_high labels."""

    def __init__(
        self,
        idx_pos: np.ndarray,
        idx_neg: np.ndarray,
        batch_size: int,
        seed: int = 42,
    ):
        assert batch_size % 5 == 0, "batch_size must be divisible by 5 for 1:4 pos:neg"
        self.idx_pos = np.asarray(idx_pos, dtype=np.int64)
        self.idx_neg = np.asarray(idx_neg, dtype=np.int64)
        self.batch_size = batch_size
        self.seed = seed

        if len(self.idx_pos) == 0:
            raise RuntimeError("No positive samples (js_high) in training pool.")
        if len(self.idx_neg) == 0:
            raise RuntimeError("No negative samples in training pool.")

        self.n_pos = batch_size // 5
        self.n_neg = batch_size - self.n_pos
        self._call = 0

        self._n_batches = max(
            1,
            len(self.idx_pos) // self.n_pos,
            len(self.idx_neg) // max(self.n_neg, 1),
        )

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._call)
        self._call += 1
        for _ in range(self._n_batches):
            pos_b = rng.choice(
                self.idx_pos, size=self.n_pos, replace=len(self.idx_pos) < self.n_pos
            )
            neg_b = rng.choice(
                self.idx_neg, size=self.n_neg, replace=len(self.idx_neg) < self.n_neg
            )
            batch = np.concatenate([pos_b, neg_b])
            rng.shuffle(batch)
            yield batch.tolist()


def materialize_router_rows(
    run_index: np.ndarray,
    token_pos: np.ndarray,
    router_repr_dir: Path,
    hidden_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One NPZ load per run; fill per-row last_hidden_state and routing_input_token_ids.
    Returns (hidden_states [N,H] float32, token_ids [N] int64).
    """
    router_repr_dir = Path(router_repr_dir)
    n = len(run_index)
    order = np.argsort(run_index, kind="mergesort")
    ri_sorted = run_index[order]
    pos_sorted = token_pos[order]
    hid_out = None
    tok_out = np.zeros(n, dtype=np.int64)

    i = 0
    pbar = tqdm(total=n, desc="materialize router_repr rows")
    while i < n:
        rid = int(ri_sorted[i])
        j = i
        while j < n and int(ri_sorted[j]) == rid:
            j += 1
        path = router_repr_dir / f"run_{rid:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing router_repr: {path}")
        z = np.load(path, mmap_mode="r")
        H = z["last_hidden_state"]
        T = z["routing_input_token_ids"]
        if hidden_dim is None:
            hidden_dim = int(H.shape[1])
            hid_out = np.zeros((n, hidden_dim), dtype=np.float32)
        block_idx = order[i:j]
        local_pos = pos_sorted[i:j].astype(np.int64)
        hid_out[block_idx] = np.asarray(H[local_pos], dtype=np.float32)
        tok_out[block_idx] = np.asarray(T[local_pos], dtype=np.int64)
        pbar.update(j - i)
        i = j
    pbar.close()
    assert hid_out is not None
    return hid_out, tok_out


class RouterMultiTaskDataset(Dataset):
    """All per-row tensors in memory (fast random access for stratified batches)."""

    def __init__(
        self,
        logits100: np.ndarray,
        hidden_states: np.ndarray,
        token_ids: np.ndarray,
        js: np.ndarray,
        entropy: np.ndarray,
        label: np.ndarray,
    ):
        self.logits100 = logits100
        self.hidden_states = hidden_states
        self.token_ids = token_ids
        self.js = js
        self.entropy = entropy
        self.label = label

    def __len__(self):
        return len(self.js)

    def __getitem__(self, i: int):
        return {
            "logits": torch.from_numpy(np.asarray(self.logits100[i], dtype=np.float32)),
            "hidden_states": torch.from_numpy(np.asarray(self.hidden_states[i], dtype=np.float32)),
            "token": torch.tensor(int(self.token_ids[i]), dtype=torch.long),
            "js": torch.tensor(float(self.js[i]), dtype=torch.float32),
            "entropy": torch.tensor(float(self.entropy[i]), dtype=torch.float32),
            "label": torch.tensor(int(self.label[i]), dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "logits": torch.stack([b["logits"] for b in batch]),
        "hidden_states": torch.stack([b["hidden_states"] for b in batch]),
        "token": torch.stack([b["token"] for b in batch]),
        "js": torch.stack([b["js"] for b in batch]),
        "entropy": torch.stack([b["entropy"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def load_split_arrays(split_path: Path, name: str = "split"):
    ds = load_from_disk(str(split_path))
    n = len(ds)
    print(f"Loading {n} rows ({name}) into memory...")
    js = np.asarray(ds["js_value"], dtype=np.float32)
    entropy = np.asarray(ds["entropy_value"], dtype=np.float32)
    label = (js > 0.1).astype(np.int64)
    run_index = np.asarray(ds["run_index"], dtype=np.int64)
    token_pos = np.asarray(ds["token_pos"], dtype=np.int32)
    sl = np.asarray(ds["small_logits"], dtype=np.float32)
    if sl.ndim != 2 or sl.shape[1] < 100:
        raise ValueError(f"Expected small_logits (N, >=100), got {sl.shape}")
    logits100 = sl[:, :100].copy()

    return logits100, js, entropy, label, run_index, token_pos


def build_strata_indices(label: np.ndarray):
    pos = np.where(label == 1)[0]
    neg = np.where(label == 0)[0]
    return pos, neg


def bce_pos_weight_from_label(label: np.ndarray, power: float = 0.5) -> float:
    """Down-weight focal imbalance: pos_weight = (neg/pos)**power (power=1 is full ratio)."""
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    if n_pos == 0:
        return 1.0
    ratio = n_neg / n_pos
    return float(ratio**power)


def lr_cosine_warmup(
    epoch: int, total_epochs: int, base_lr: float, warmup_epochs: int, eta_min_ratio: float = 0.01
) -> float:
    """Linear warmup then cosine decay to eta_min_ratio * base_lr."""
    if total_epochs <= 0:
        return base_lr
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(warmup_epochs, 1))
    t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    t = min(max(t, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    eta_min = base_lr * eta_min_ratio
    return eta_min + (base_lr - eta_min) * cos


def best_threshold_f1(y_true: np.ndarray, probs: np.ndarray, n_grid: int = 181) -> tuple[float, float]:
    """Grid-search probability threshold for maximum binary F1 (JS > 0.1)."""
    y = y_true.astype(np.int32)
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, n_grid):
        pred = (probs >= t).astype(np.int32)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = float(f), float(t)
    return best_t, best_f1


def best_threshold_recall_constrained(
    y_true: np.ndarray, probs: np.ndarray, min_recall: float = 0.90, n_grid: int = 181
) -> tuple[float, float, float, float]:
    """Find threshold with lowest positive rate subject to recall >= min_recall."""
    y = y_true.astype(np.int32)
    best_t, best_recall, best_pos_rate, best_f1 = 0.5, 0.0, 1.0, 0.0
    for t in np.linspace(0.01, 0.99, n_grid):
        pred = (probs >= t).astype(np.int32)
        rec = recall_score(y, pred, pos_label=1, zero_division=0)
        if rec < min_recall:
            continue
        pos_rate = pred.mean()
        if pos_rate < best_pos_rate:
            best_pos_rate = pos_rate
            best_recall = rec
            best_t = float(t)
            best_f1 = float(f1_score(y, pred, zero_division=0))
    return best_t, best_recall, best_pos_rate, best_f1


@torch.no_grad()
def eval_probs(model, loader, device):
    model.eval()
    probs, labels, entropies = [], [], []
    for batch in loader:
        logits = batch["logits"].to(device)
        hid = batch["hidden_states"].to(device)
        tok = batch["token"].to(device)
        js = batch["js"].to(device)
        cls_logits, _ = model(logits, hid, tok)
        p = torch.sigmoid(cls_logits.squeeze(-1))
        probs.append(p.cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        entropies.append(batch["entropy"].numpy())
    return (
        np.concatenate(probs),
        np.concatenate(labels),
        np.concatenate(entropies),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default=str(DEFAULT_DATA_ROOT / "train"))
    ap.add_argument("--val_path", type=str, default=str(DEFAULT_DATA_ROOT / "validation"))
    ap.add_argument("--router_repr_dir", type=str, default=str(DEFAULT_ROUTER_REPR))
    ap.add_argument(
        "--epochs",
        type=int,
        default=64,
        help="Max epochs; training stops earlier if val quality plateaus (early stopping).",
    )
    ap.add_argument("--batch_size", type=int, default=2560, help="divisible by 5 (1:4 sampling)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--aux_weight", type=float, default=0.2)
    ap.add_argument("--smooth_lo", type=float, default=0.08)
    ap.add_argument("--smooth_hi", type=float, default=0.12)
    ap.add_argument("--entropy_subset_thr", type=float, default=0.6)
    ap.add_argument(
        "--js_cls_threshold",
        type=float,
        default=None,
        help="If set, recompute binary labels from js_value > threshold (ignores on-disk js_high). "
        "Use 0.1 to match plot_pr_curves scripts; default None keeps dataset js_high (often from tau_js in manifest).",
    )
    ap.add_argument(
        "--verify_entropy_proxy",
        action="store_true",
        help="Check entropy_value vs Shannon(top-100 logits) on train split (subset).",
    )
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap on train rows after materialize (for smoke runs).",
    )
    ap.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Optional cap on validation rows.",
    )
    ap.add_argument(
        "--counterfactual_jsonl",
        type=str,
        default=None,
        help="Optional JSONL with keys run_index, token_pos, label in [0,1]; merges into js_high (see r2r.utils.counterfactual_router_labels).",
    )
    ap.add_argument("--out_dir", type=str, default="/root/autodl-tmp/datasets/router_multitask_js_runs/run_default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--model_type",
        type=str,
        default="ffn4",
        choices=["ffn4", "bottleneck2", "bottleneck3", "bottleneck3_v7", "bottleneck3_v8", "bottleneck_r2r"],
        help="Router architecture: ffn4, bottleneck2/3, v7(+entropy), v8(dual-head), bottleneck_r2r(r2r-style).",
    )
    ap.add_argument("--ffn_dim", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.12)
    ap.add_argument("--no_input_layernorm", action="store_true")
    ap.add_argument(
        "--freeze_token_embeddings",
        action="store_true",
        help="If set, keep 1.5B token embedding copy frozen (default: train embedding as part of input).",
    )
    ap.add_argument("--bce_pos_weight_power", type=float, default=0.5)
    ap.add_argument(
        "--warmup_epoch_frac",
        type=float,
        default=0.08,
        help="Linear LR warmup for this fraction of total epochs, then cosine decay.",
    )
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=8,
        help="Stop if neither val PR-AUC nor val F1 improves (by min-deltas) for this many epochs.",
    )
    ap.add_argument(
        "--early_stop_min_delta_pr",
        type=float,
        default=1e-4,
        help="PR-AUC change below this does not count as improvement.",
    )
    ap.add_argument(
        "--early_stop_min_delta_f1",
        type=float,
        default=5e-4,
        help="Val F1 change below this does not count as improvement.",
    )
    ap.add_argument(
        "--checkpoint_score_f1_weight",
        type=float,
        default=0.05,
        help="When saving best.pt, maximize val_pr_auc + w * val_f1 (tie-break composite).",
    )
    ap.add_argument("--aux_focal_frac", type=float, default=0.5, help="aux = (1-a)*MSE + a*FocalMSE")
    ap.add_argument("--focal_gamma", type=float, default=0.0,
        help="Focal BCE gamma for classification head (0=plain BCE, 2.0 typical).")
    ap.add_argument("--div_weight", type=float, default=1.0,
        help="Weight for divergent auxiliary head loss (bottleneck3_v8 only).")
    ap.add_argument("--entropy_weight", type=float, default=0.0,
        help="Weight multiplier for low-entropy tokens in BCE loss (0=disabled, 3.0 suggested).")
    ap.add_argument("--lowent_downsample", type=float, default=0.0,
        help="Entropy threshold for low-entropy negative downsampling (0=disabled, 0.6 suggested).")
    ap.add_argument("--lowent_downsample_ratio", type=int, default=10,
        help="Target pos:neg ratio in low-entropy region (default=10 means 1:10).")
    ap.add_argument("--lowent_only", action="store_true",
        help="Only use low-entropy data (drop high-entropy region entirely).")
    ap.add_argument("--lowent_cat_map", type=str, default='/tmp/run2cat.npz',
        help="NPZ with run_indices and categories for low-entropy downsampling.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repr_dir = Path(args.router_repr_dir)

    logits100, js, entropy, js_high_tr, run_index, token_pos = load_split_arrays(
        Path(args.train_path), "train"
    )
    if args.js_cls_threshold is not None:
        js_high_tr = js_high_from_js_values(js, args.js_cls_threshold)
        print(
            f"Using recomputed js_high (js > {args.js_cls_threshold}): "
            f"pos_rate={float(js_high_tr.mean()):.4f}"
        )
    if args.counterfactual_jsonl:
        cf = load_counterfactual_map(Path(args.counterfactual_jsonl))
        if cf:
            before = float(js_high_tr.mean())
            js_high_tr = merge_js_high_with_counterfactual(
                run_index, token_pos, js_high_tr, cf, cf_weight=1.0, cf_threshold=0.5
            )
            print(
                f"Merged counterfactual_jsonl ({len(cf)} keys): js_high pos_rate {before:.4f} -> {float(js_high_tr.mean()):.4f}"
            )
        else:
            print("counterfactual_jsonl path set but no rows loaded; skipping merge.")

    if args.verify_entropy_proxy:
        n_chk = min(50_000, len(logits100))
        pred_ent = entropy_topk_shannon_logits(logits100[:n_chk])
        if pred_ent.ndim == 0:
            pred_ent = np.asarray([float(pred_ent)])
        err = np.abs(pred_ent - entropy[:n_chk])
        mx = float(err.max())
        print(f"[verify_entropy_proxy] max_abs_err on {n_chk} rows: {mx:.6f}")
        if mx > 0.02:
            raise RuntimeError(
                "entropy_value does not match top-100 Shannon on logits100; "
                "check dataset small_logits layout (first 100 cols must be raw top_logits_1p5b)."
            )

    if args.max_train_samples is not None and args.max_train_samples < len(js):
        n = args.max_train_samples
        logits100, js, entropy, js_high_tr, run_index, token_pos = (
            logits100[:n],
            js[:n],
            entropy[:n],
            js_high_tr[:n],
            run_index[:n],
            token_pos[:n],
        )
        print(f"Capped train split to max_train_samples={n}")

    pos_i, neg_i = build_strata_indices(js_high_tr)
    print(f"Strata counts: pos={len(pos_i)}, neg={len(neg_i)}")

    if args.lowent_downsample:
        lowent_thr = args.lowent_downsample
        le = entropy < lowent_thr
        le_pos = np.where((js_high_tr > 0.5) & le)[0]
        le_neg = np.where((js_high_tr < 0.5) & le)[0]
        he_neg = np.where((js_high_tr < 0.5) & ~le)[0]
        target_le_neg = len(le_pos) * args.lowent_downsample_ratio
        # Category sampling
        cat_map = dict(np.load(args.lowent_cat_map, allow_pickle=True).items()) if args.lowent_cat_map else {}
        if cat_map:
            run2cat = {int(r): str(c) for r, c in zip(cat_map['run_indices'], cat_map['categories'])}
            token_cats = np.array([run2cat.get(int(r), 'MATH') for r in run_index])
        else:
            token_cats = np.array(['MATH'] * len(run_index))
        ratios = {'MATH': 0.4, 'CODE': 0.3, 'QA': 0.3}
        rng = np.random.default_rng(args.seed)
        sampled_le_neg = []
        for cat, ratio in ratios.items():
            cat_le_neg = le_neg[token_cats[le_neg] == cat]
            n_target = int(target_le_neg * ratio)
            n_sample = min(n_target, len(cat_le_neg))
            if n_sample > 0:
                sampled = rng.choice(cat_le_neg, size=n_sample, replace=False)
                sampled_le_neg.append(sampled)
            print(f"  {cat}: lowEnt-neg {len(cat_le_neg):,} → sampled {n_sample:,}")
        sampled_le_neg = np.concatenate(sampled_le_neg).astype(np.int64)
        if args.lowent_only:
            # Only low-entropy region: all positives + downsampled negatives
            pos_i = le_pos.astype(np.int64)
            neg_i = sampled_le_neg
            print(f"LowEnt-only: pos={len(pos_i):,}, neg={len(neg_i):,}")
        else:
            neg_i = np.concatenate([sampled_le_neg, he_neg]).astype(np.int64)
            print(f"Downsampled: pos={len(pos_i):,}, lowEnt-neg={len(sampled_le_neg):,}, highEnt-neg={len(he_neg):,}, total-neg={len(neg_i):,}")

    print("Materializing train hidden/token from router_repr (one-time)...")
    hid_tr, tok_tr = materialize_router_rows(run_index, token_pos, repr_dir)

    train_ds = RouterMultiTaskDataset(logits100, hid_tr, tok_tr, js, entropy, js_high_tr)
    batch_sampler = StratifiedRouterBatchSampler(
        pos_i, neg_i, batch_size=args.batch_size, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Validation (sequential; full pass)
    vpath = Path(args.val_path)
    v_logits100, v_js, v_entropy, v_div, v_run, v_tok = load_split_arrays(vpath, "validation")
    if args.js_cls_threshold is not None:
        v_div = js_high_from_js_values(v_js, args.js_cls_threshold)
        print(
            f"Validation js_high recomputed (js > {args.js_cls_threshold}): "
            f"pos_rate={float(v_div.mean()):.4f}"
        )
    if args.counterfactual_jsonl:
        cf = load_counterfactual_map(Path(args.counterfactual_jsonl))
        if cf:
            v_div = merge_js_high_with_counterfactual(v_run, v_tok, v_div, cf)

    if args.max_val_samples is not None and args.max_val_samples < len(v_js):
        n = args.max_val_samples
        v_logits100, v_js, v_entropy, v_div, v_run, v_tok = (
            v_logits100[:n],
            v_js[:n],
            v_entropy[:n],
            v_div[:n],
            v_run[:n],
            v_tok[:n],
        )
        print(f"Capped validation split to max_val_samples={n}")
    print("Materializing validation hidden/token...")
    hid_va, tok_va = materialize_router_rows(v_run, v_tok, repr_dir)
    val_ds = RouterMultiTaskDataset(v_logits100, hid_va, tok_va, v_js, v_entropy, v_div)
    val_loader = DataLoader(
        val_ds,
        batch_size=min(4096, len(val_ds)),
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    pw = bce_pos_weight_from_label(js_high_tr, power=args.bce_pos_weight_power)
    print(f"BCE pos_weight (power={args.bce_pos_weight_power}): {pw:.4f}")

    print(
        "Token embedding: "
        + ("frozen (1.5B copy)" if args.freeze_token_embeddings else "trainable (in fusion path)")
    )

    if args.model_type == "bottleneck2":
        model = RouterBottleneck2Block(
            dropout=args.dropout,
            normalize_inputs=not args.no_input_layernorm,
            freeze_token_embeddings=args.freeze_token_embeddings,
            pretrained_model_name=MODEL15_PATH,
        ).to(device)
    elif args.model_type == "bottleneck3":
        model = RouterBottleneck3Block(
            dropout=args.dropout,
            normalize_inputs=not args.no_input_layernorm,
            freeze_token_embeddings=args.freeze_token_embeddings,
            pretrained_model_name=MODEL15_PATH,
        ).to(device)
    elif args.model_type == "bottleneck3_v7":
        model = RouterBottleneck3BlockV7(
            dropout=args.dropout,
            normalize_inputs=not args.no_input_layernorm,
            freeze_token_embeddings=args.freeze_token_embeddings,
            pretrained_model_name=MODEL15_PATH,
        ).to(device)
    elif args.model_type == "bottleneck3_v8":
        model = RouterBottleneck3BlockV8(
            dropout=args.dropout,
            normalize_inputs=not args.no_input_layernorm,
            freeze_token_embeddings=args.freeze_token_embeddings,
            pretrained_model_name=MODEL15_PATH,
        ).to(device)
    elif args.model_type == "bottleneck_r2r":
        model = RouterBottleneckR2R(
            dropout=args.dropout,
            normalize_inputs=not args.no_input_layernorm,
            freeze_token_embeddings=args.freeze_token_embeddings,
            pretrained_model_name=MODEL15_PATH,
        ).to(device)
    else:
        model = RouterMultiTaskFFN4(
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            normalize_inputs=not args.no_input_layernorm,
            freeze_token_embeddings=args.freeze_token_embeddings,
            pretrained_model_name=MODEL15_PATH,
        ).to(device)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "ln_" in name or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    focal_mse = FocalMSELoss(gamma=2.0)
    pos_w = torch.tensor([pw], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    warmup_epochs = max(1, int(round(args.epochs * args.warmup_epoch_frac)))

    best_composite = -1.0e9
    stall = 0
    for epoch in range(args.epochs):
        lr_ep = lr_cosine_warmup(epoch, args.epochs, args.lr, warmup_epochs)
        for g in opt.param_groups:
            g["lr"] = lr_ep

        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            logits = batch["logits"].to(device)
            hid = batch["hidden_states"].to(device)
            tok = batch["token"].to(device)
            js = batch["js"].to(device)

            ent = batch["entropy"].to(device) if args.model_type == "bottleneck3_v7" else None
            if args.model_type == "bottleneck3_v7":
                cls_logits, reg_pred = model(logits, hid, tok, entropy=ent)
            else:
                cls_logits, reg_pred = model(logits, hid, tok)
            if args.model_type == "bottleneck3_v8":
                # V8: dual head, second output is divergent logits
                cls_logits, div_logits = cls_logits, reg_pred
            y_div = batch["label"].to(device)
            # Compute per-token weight for entropy weighting
            if args.entropy_weight > 0:
                ent_val = batch["entropy"].to(device)
                token_weight = 1.0 + args.entropy_weight * (ent_val < 0.6).float()
            else:
                token_weight = None
            if args.focal_gamma > 0:
                bce_each = nn.functional.binary_cross_entropy_with_logits(
                    cls_logits.squeeze(-1), y_div, reduction='none')
                prob = torch.sigmoid(cls_logits.squeeze(-1))
                p_t = torch.where(y_div > 0.5, prob, 1.0 - prob)
                focal_weight = (1.0 - p_t).pow(args.focal_gamma)
                weight = focal_weight * (token_weight if token_weight is not None else 1.0)
                loss_cls = (weight * bce_each).mean()
            else:
                if token_weight is not None:
                    bce_each = nn.functional.binary_cross_entropy_with_logits(
                        cls_logits.squeeze(-1), y_div, reduction='none')
                    loss_cls = (token_weight * bce_each).mean()
                else:
                    loss_cls = bce(cls_logits.squeeze(-1), y_div)
            loss = loss_cls
            if args.model_type == "bottleneck3_v8" and args.div_weight > 0:
                # Divergent head: JS>0.1 AND entropy<0.6
                ent_val = batch["entropy"].to(device)
                y_div = batch["label"].to(device)
                y_divergent = (y_div > 0.5) & (ent_val < 0.6)
                y_divergent = y_divergent.float()
                # pos_weight: neg/pos for this extremely imbalanced label
                n_div = y_divergent.numel()
                n_pos = y_divergent.sum().clamp_min(1)
                div_pw = (n_div - n_pos) / n_pos
                div_bce = nn.BCEWithLogitsLoss(pos_weight=div_pw)
                loss_div = div_bce(div_logits.squeeze(-1), y_divergent)
                loss = loss + args.div_weight * loss_div
            if args.aux_weight > 0:
                y_js = batch["js"].to(device)
                rp = reg_pred.squeeze(-1)
                mse = F.mse_loss(rp, y_js)
                focal = focal_mse(rp, y_js)
                loss_reg = (1.0 - args.aux_focal_frac) * mse + args.aux_focal_frac * focal
                loss = loss + args.aux_weight * loss_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(loss.item())

        probs, y_true, ent = eval_probs(model, val_loader, device)
        prec, rec, thr = precision_recall_curve(y_true, probs)
        pr_auc = auc(rec, prec)
        best_t, val_recall, val_pos_rate, val_f1 = best_threshold_recall_constrained(
            y_true, probs, min_recall=0.90
        )
        low_ent = ent < args.entropy_subset_thr
        if np.any(low_ent) and np.any(y_true[low_ent] > 0):
            y_sub = y_true[low_ent].astype(np.int32)
            rec_sub_05 = recall_score(
                y_sub,
                (probs >= 0.5)[low_ent].astype(np.int32),
                pos_label=1,
                zero_division=0,
            )
            rec_sub_bt = recall_score(
                y_sub,
                (probs >= best_t)[low_ent].astype(np.int32),
                pos_label=1,
                zero_division=0,
            )
        else:
            rec_sub_05 = rec_sub_bt = float("nan")

        print(
            f"epoch {epoch+1}: lr={lr_ep:.2e} train_loss={np.mean(losses):.5f} val_pr_auc={pr_auc:.5f} "
            f"val_recall@rc90={val_recall:.4f} pos_rate={val_pos_rate:.4f} thr={best_t:.3f} "
            f"lowEnt_recall@0.5={rec_sub_05:.4f} lowEnt_recall@{best_t:.2f}={rec_sub_bt:.4f}"
        )

        # Composite: prefer lower positive_rate at same recall level.
        # For lowent_only, prioritize low-entropy recall instead of PR-AUC.
        if args.lowent_only:
            composite = rec_sub_05 - args.checkpoint_score_f1_weight * val_pos_rate
        else:
            composite = pr_auc - args.checkpoint_score_f1_weight * val_pos_rate
        improved = composite > best_composite + 1e-7

        if improved:
            best_composite = composite
            stall = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_pr_auc": pr_auc,
                    "val_recall_at_rc90": val_recall,
                    "best_prob_threshold": best_t,
                    "composite_score": composite,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )
        else:
            stall += 1

        if stall >= args.early_stop_patience:
            print(
                f"Early stop: no composite improvement for {args.early_stop_patience} "
                f"consecutive epochs (best composite={best_composite:.5f})."
            )
            break

    # Load best for final plots / metrics
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    probs, y_true, ent = eval_probs(model, val_loader, device)

    prec, rec, thr = precision_recall_curve(y_true, probs)
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    thr_lbl = args.js_cls_threshold if args.js_cls_threshold is not None else "disk_js_high"
    plt.title(f"Validation PR (js_high @ {thr_lbl})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve_validation.png", dpi=150)
    plt.close()

    best_thr, val_recall_final, val_pos_rate_final, val_f1_final = best_threshold_recall_constrained(
        y_true, probs, min_recall=0.90
    )
    cls_thr = 0.5
    low_ent = ent < args.entropy_subset_thr
    rec_sub_05 = rec_sub_best = float("nan")
    if np.any(low_ent) and np.any(y_true[low_ent] > 0):
        y_sub = y_true[low_ent].astype(np.int32)
        rec_sub_05 = float(
            recall_score(
                y_sub,
                (probs >= cls_thr)[low_ent].astype(np.int32),
                pos_label=1,
                zero_division=0,
            )
        )
        rec_sub_best = float(
            recall_score(
                y_sub,
                (probs >= best_thr)[low_ent].astype(np.int32),
                pos_label=1,
                zero_division=0,
            )
        )

    metrics = {
        "val_pr_auc": float(pr_auc),
        "val_f1_at_best_threshold": float(val_f1_final),
        "val_recall_at_best_threshold": float(val_recall_final),
        "val_positive_rate_at_best_threshold": float(val_pos_rate_final),
        "best_prob_threshold_on_val": float(best_thr),
        "optimization_target": "recall>=0.90_min_positive_rate",
        f"recall_js_high_given_entropy_lt_{args.entropy_subset_thr}_thr_0.5": rec_sub_05,
        f"recall_js_high_given_entropy_lt_{args.entropy_subset_thr}_thr_best_val_f1": rec_sub_best,
        "n_val": int(len(y_true)),
        "n_val_low_entropy": int(low_ent.sum()),
        "best_epoch": int(ckpt["epoch"]),
        "js_cls_threshold": args.js_cls_threshold,
        "aux_weight": float(args.aux_weight),
        "counterfactual_jsonl": args.counterfactual_jsonl,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

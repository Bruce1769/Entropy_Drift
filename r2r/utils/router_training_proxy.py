"""
Shared definitions for multitask JS router *training data* vs *inference* (MultitaskJSRouterSwitching).

Keep numpy/torch helpers here so dataset builders and train_router_multitask_js.py stay aligned
with [r2r/utils/switching.py] MultitaskJSRouterSwitching.route:
  - entropy: Shannon on softmax(last-step top-k logits), k=100 by default.
  - js_value: comes from offline NPZ (prefill union-support JS); not recomputed in training loop.
"""

from __future__ import annotations

import numpy as np
import torch


def entropy_topk_shannon_logits(
    logits_topk: np.ndarray | torch.Tensor,
    eps: float = 1e-12,
) -> np.ndarray | torch.Tensor:
    """
    Match MultitaskJSRouterSwitching.route:
      probs = softmax(logits_topk, dim=-1)
      entropy = -(probs * log(probs)).sum(-1)
    """
    if isinstance(logits_topk, torch.Tensor):
        probs = torch.softmax(logits_topk.to(dtype=torch.float32), dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
        return ent
    x = logits_topk.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    p = (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float64)
    return (-(p * np.log(p + eps)).sum(axis=-1)).astype(np.float32)


def js_high_from_js_values(js: np.ndarray, threshold: float) -> np.ndarray:
    """Binary label (int64 0/1) for JS above threshold."""
    return (js.astype(np.float32) > float(threshold)).astype(np.int64)


def verify_entropy_numpy_matches_torch(logits_2d: np.ndarray, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """Sanity check for unit tests / verify_router_proxy_alignment."""
    a = entropy_topk_shannon_logits(logits_2d)
    t = torch.from_numpy(logits_2d.astype(np.float32))
    b = entropy_topk_shannon_logits(t).numpy()
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))

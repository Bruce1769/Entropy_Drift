"""
Skeleton for counterfactual routing labels: "would escalating to 32B at this prefix help the final answer?"

Offline pipeline (to implement per your rollout budget):
  1. For each (run_index, token_pos), you have router_repr + logits from NPZ / js_entropy_risk_dataset.
  2. Branch A: continue with 1.5B greedy from prefix; Branch B: switch to 32B from same prefix.
  3. Compare final extracted answers (or token-level NLL of reference completion).

This module only defines the merge contract for train_router_multitask_js when a sidecar file exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class CounterfactualRow:
    run_index: int
    token_pos: int
    label: float  # 1.0 = 32B branch strictly better, 0.0 = not better, optional soft in (0,1)


def iter_counterfactual_jsonl(path: Path) -> Iterator[CounterfactualRow]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            yield CounterfactualRow(
                run_index=int(o["run_index"]),
                token_pos=int(o["token_pos"]),
                label=float(o["label"]),
            )


def load_counterfactual_map(path: Path | None) -> dict[tuple[int, int], float]:
    if path is None or not path.is_file():
        return {}
    out: dict[tuple[int, int], float] = {}
    for row in iter_counterfactual_jsonl(path):
        out[(row.run_index, row.token_pos)] = row.label
    return out


def merge_js_high_with_counterfactual(
    run_index: np.ndarray,
    token_pos: np.ndarray,
    js_high: np.ndarray,
    cf_map: dict[tuple[int, int], float],
    cf_weight: float = 1.0,
    cf_threshold: float = 0.5,
) -> np.ndarray:
    """
    If a counterfactual label exists for a row, set js_high = 1 when label >= cf_threshold.
    Rows without CF labels keep the original js_high.
    """
    if not cf_map:
        return js_high
    out = js_high.astype(np.int64).copy()
    for i in range(len(out)):
        key = (int(run_index[i]), int(token_pos[i]))
        if key in cf_map:
            v = cf_map[key]
            if v >= cf_threshold:
                out[i] = 1
            elif cf_weight >= 1.0 and v < cf_threshold:
                out[i] = 0
    return out


def write_example_skeleton(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        {"run_index": 0, "token_pos": 10, "label": 1.0, "note": "32B fixes boxed answer vs 1.5B"},
        {"run_index": 0, "token_pos": 11, "label": 0.0, "note": "no gain"},
    ]
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

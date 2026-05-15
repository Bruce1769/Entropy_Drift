#!/usr/bin/env python3
"""Write example counterfactual JSONL for merge with train_router_multitask_js --counterfactual_jsonl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from r2r.utils.counterfactual_router_labels import write_example_skeleton  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/root/autodl-tmp/datasets/counterfactual_router_example.jsonl")
    args = ap.parse_args()
    write_example_skeleton(Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPECIAL_RE = re.compile(r"^<.*>$")
PUNCT_RE = re.compile(r"^[\W_]+$")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")


def normalize_token(token: str) -> str:
    if pd.isna(token):
        return ""
    token = str(token).replace("\r\n", "\n")
    if token.strip() == "":
        return ""
    token = token.replace("\n", "\\n").strip()
    return token


def keep_token(token: str) -> bool:
    if not token:
        return False
    if SPECIAL_RE.match(token):
        return False
    if PUNCT_RE.match(token):
        return False
    if NUMBER_RE.match(token):
        return False
    return True


def make_wordcloud(freqs: Counter, title: str, output_path: Path) -> None:
    items = freqs.most_common(120)
    counts = np.array([count for _, count in items], dtype=float)
    min_size, max_size = 12.0, 46.0
    if counts.max() == counts.min():
        sizes = np.full_like(counts, (min_size + max_size) / 2.0)
    else:
        sizes = min_size + (counts - counts.min()) / (counts.max() - counts.min()) * (max_size - min_size)

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("white")
    cmap = plt.get_cmap("viridis")

    placed = []
    for idx, ((token, _), size) in enumerate(zip(items, sizes)):
        width = min(0.45, 0.011 * len(token) * (size / max_size) + 0.02)
        height = min(0.12, 0.055 * (size / max_size) + 0.01)
        for _ in range(250):
            x = rng.uniform(width / 2, 1 - width / 2)
            y = rng.uniform(height / 2, 1 - height / 2)
            overlaps = any(abs(x - px) < (width + pw) / 2 and abs(y - py) < (height + ph) / 2 for px, py, pw, ph in placed)
            if not overlaps:
                placed.append((x, y, width, height))
                ax.text(
                    x,
                    y,
                    token,
                    fontsize=float(size),
                    color=cmap(idx / max(len(items) - 1, 1)),
                    ha="center",
                    va="center",
                    rotation=int(rng.choice([0, 0, 0, 15, -15])),
                    alpha=0.95,
                )
                break

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate word clouds for threshold quadrants.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quadrant-col", default="threshold_quadrant_name")
    parser.add_argument("--token-col", default="output_token_str_online")
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, usecols=[args.quadrant_col, args.token_col])
    df["quadrant"] = df[args.quadrant_col].astype(str)
    df["token"] = df[args.token_col].map(normalize_token)
    df = df[df["token"].map(keep_token)]

    top_rows = []
    for quadrant, sub in df.groupby("quadrant"):
        freqs = Counter(sub["token"])
        if not freqs:
            continue
        make_wordcloud(freqs, f"{quadrant} Word Cloud", output_dir / f"{quadrant}_wordcloud.png")
        for token, count in freqs.most_common(args.top_n):
            top_rows.append({"quadrant": quadrant, "token": token, "count": count})

    pd.DataFrame(top_rows).to_csv(output_dir / "threshold_quadrant_wordcloud_top_tokens.csv", index=False)
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

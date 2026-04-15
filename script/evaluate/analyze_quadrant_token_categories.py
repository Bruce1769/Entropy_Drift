#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RE_SPECIAL = re.compile(r"^<.*>$")
RE_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
RE_PUNCT = re.compile(r"^[\.,;:!\?\"'`“”‘’\(\)\[\]\{\}]+$")
RE_MATH = re.compile(r"^[=\+\-\*/\^%$<>≤≥×÷±∓≈≠∈∉∑∏√∞∠⊂⊆⇒→←]+$")
RE_ALPHA = re.compile(r"^[A-Za-z]+$")
RE_ALNUM = re.compile(r"^[A-Za-z0-9_]+$")

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "being", "been", "that", "this", "it",
    "as", "at", "by", "from", "have", "has", "had", "can", "could", "should",
    "would", "if", "then", "than", "because", "since", "so", "but", "also",
    "we", "i", "you", "they", "he", "she", "there", "their", "our", "my",
    "me", "us", "not", "no", "do", "does", "did", "let",
}

REASONING_MARKERS = {
    "okay", "wait", "therefore", "thus", "hence", "suppose", "assume",
    "note", "alternatively", "first", "next", "finally", "now", "maybe",
    "perhaps", "denote", "consider", "compute", "check", "trying", "try",
    "need", "think", "means", "gives", "shows", "implies", "however",
}


def normalize_token(token: str) -> str:
    if pd.isna(token):
        return ""
    token = str(token).replace("\r\n", "\n")
    if token == " ":
        return "SPACE"
    if token == "\n":
        return "NEWLINE"
    if token == "\n\n":
        return "DOUBLE_NEWLINE"
    if token.strip() == "":
        if "\n" in token:
            return f"{token.count(chr(10))}xNEWLINE"
        return "SPACE"
    return token


def classify_token(token: str) -> str:
    token = normalize_token(token)
    stripped = token.strip()
    lowered = stripped.lower()

    if not stripped:
        return "whitespace"
    if token in {"SPACE", "NEWLINE", "DOUBLE_NEWLINE"} or token.endswith("xNEWLINE"):
        return "whitespace"
    if RE_SPECIAL.match(stripped):
        return "special_token"
    if RE_NUMBER.match(stripped):
        return "number"
    if RE_MATH.match(stripped):
        return "math_symbol"
    if RE_PUNCT.match(stripped):
        return "punctuation"
    if lowered in REASONING_MARKERS:
        return "reasoning_marker"
    if lowered in FUNCTION_WORDS:
        return "function_word"
    if RE_ALPHA.match(stripped):
        if stripped[0].islower() and not token.startswith(" "):
            return "subword_fragment"
        return "content_word"
    if RE_ALNUM.match(stripped):
        if any(ch.isdigit() for ch in stripped) and any(ch.isalpha() for ch in stripped):
            return "alnum_token"
        if stripped[0].islower() and not token.startswith(" "):
            return "subword_fragment"
        return "content_word"
    return "mixed_token"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_grouped_bars(counts: pd.DataFrame, output_path: Path, value_col: str, title: str) -> None:
    quadrants = list(counts["quadrant"].drop_duplicates())
    categories = list(counts["category"].drop_duplicates())
    pivot = (
        counts.pivot(index="category", columns="quadrant", values=value_col)
        .reindex(index=categories, columns=quadrants)
        .fillna(0.0)
    )
    x = np.arange(len(categories))
    width = 0.18
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    for idx, quadrant in enumerate(quadrants):
        ax.bar(
            x + (idx - 1.5) * width,
            pivot[quadrant].to_numpy(),
            width=width,
            label=quadrant,
            color=colors[idx % len(colors)],
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_heatmap(counts: pd.DataFrame, output_path: Path, value_col: str, title: str) -> None:
    quadrants = list(counts["quadrant"].drop_duplicates())
    categories = list(counts["category"].drop_duplicates())
    pivot = (
        counts.pivot(index="category", columns="quadrant", values=value_col)
        .reindex(index=categories, columns=quadrants)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(quadrants)))
    ax.set_xticklabels(quadrants, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify quadrant tokens and plot category distributions.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quadrant-col", default="quadrant")
    parser.add_argument("--token-col", default="output_token_str_online")
    parser.add_argument("--output-prefix", default="quadrant")
    parser.add_argument("--title-prefix", default="Quadrant")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = pd.read_csv(args.input_csv)
    df["quadrant"] = df[args.quadrant_col].astype(str)
    df["token"] = df[args.token_col].map(normalize_token)
    df["category"] = df[args.token_col].map(classify_token)

    classified_path = output_dir / f"{args.output_prefix}_token_categories.csv"
    df.to_csv(classified_path, index=False)

    counts = (
        df.groupby(["quadrant", "category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    counts["proportion"] = counts.groupby("quadrant")["count"].transform(lambda s: s / s.sum())
    counts = counts.sort_values(["quadrant", "count"], ascending=[True, False])
    counts_path = output_dir / f"{args.output_prefix}_token_category_counts.csv"
    counts.to_csv(counts_path, index=False)

    examples = (
        df.groupby(["quadrant", "category", "token"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["quadrant", "category", "count"], ascending=[True, True, False])
    )
    examples = examples.groupby(["quadrant", "category"], as_index=False).head(12)
    examples_path = output_dir / f"{args.output_prefix}_token_category_examples.csv"
    examples.to_csv(examples_path, index=False)

    plot_grouped_bars(
        counts,
        output_dir / f"{args.output_prefix}_token_category_count_distribution.png",
        "count",
        f"Token Category Count Distribution by {args.title_prefix}",
    )
    plot_grouped_bars(
        counts,
        output_dir / f"{args.output_prefix}_token_category_proportion_distribution.png",
        "proportion",
        f"Token Category Proportion Distribution by {args.title_prefix}",
    )
    plot_heatmap(
        counts,
        output_dir / f"{args.output_prefix}_token_category_proportion_heatmap.png",
        "proportion",
        f"Token Category Proportion Heatmap by {args.title_prefix}",
    )

    summary_lines = [
        f"# {args.title_prefix} Token Category Summary",
        "",
        "Categories used:",
        "- whitespace",
        "- punctuation",
        "- math_symbol",
        "- number",
        "- function_word",
        "- reasoning_marker",
        "- content_word",
        "- subword_fragment",
        "- alnum_token",
        "- mixed_token",
        "- special_token",
        "",
    ]
    for quadrant, sub in counts.groupby("quadrant"):
        top_rows = sub.sort_values("count", ascending=False).head(5)
        summary_lines.append(f"## {quadrant}")
        for _, row in top_rows.iterrows():
            summary_lines.append(
                f"- {row['category']}: count={int(row['count'])}, proportion={row['proportion']:.4f}"
            )
        summary_lines.append("")
    (output_dir / f"{args.output_prefix}_token_category_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print(f"Wrote classification outputs to {output_dir}")


if __name__ == "__main__":
    main()

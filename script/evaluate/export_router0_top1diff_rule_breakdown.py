import argparse
import json
import os
import re
from typing import Tuple

import csv
import statistics


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]*\|>")
ANGLE_TOKEN_RE = re.compile(r"<[A-Za-z_/][^>]*>")


def _norm_case_punct(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", text).lower()


def _is_special_token(text: str) -> bool:
    if not text:
        return False
    if SPECIAL_TOKEN_RE.search(text):
        return True
    if ANGLE_TOKEN_RE.search(text):
        return True
    return False


def _is_whitespace(text: str) -> bool:
    return text.strip() == ""


def _is_digits(text: str) -> bool:
    stripped = text.strip()
    return stripped.isdigit() if stripped else False


def _is_punct_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return all(not ch.isalnum() and not ch.isspace() for ch in stripped)


def _is_math_symbolic(text: str) -> bool:
    if not text:
        return False
    math_chars = set("+-*/=^<>∑√π∞≈≤≥÷×∫∏≠∈∉⊂⊆⊃⊇∪∩∧∨¬→↔")
    if any(ch in math_chars for ch in text):
        return True
    if "\\" in text or "$" in text:
        return True
    return False


def classify_pair(slm: str, llm: str) -> Tuple[str, str]:
    slm = "" if slm is None else str(slm)
    llm = "" if llm is None else str(llm)

    if _is_whitespace(slm) or _is_whitespace(llm):
        return "whitespace_formatting", "whitespace_involved"

    if _is_special_token(slm) or _is_special_token(llm):
        return "special_token_drift", "special_token_involved"

    if _is_digits(slm) and _is_digits(llm) and slm != llm:
        return "number_mismatch", "both_digits_mismatch"
    if (_is_digits(slm) and not _is_digits(llm)) or (_is_digits(llm) and not _is_digits(slm)):
        return "number_mismatch", "digit_vs_non_digit"

    if _is_math_symbolic(slm) or _is_math_symbolic(llm):
        return "math_symbol_mismatch", "math_symbol_involved"

    if _is_punct_only(slm) and _is_punct_only(llm) and slm != llm:
        return "punctuation_only", "punct_only_mismatch"

    slm_norm = _norm_case_punct(slm)
    llm_norm = _norm_case_punct(llm)
    if slm_norm and slm_norm == llm_norm and slm != llm:
        return "case_punct_only", "norm_equal_after_case_punct"

    return "lexical_semantic_substitution", "default_text_substitution"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/remote-home/pxl/R2R/output/replay_full_prefill/aime26_all30_pos8192_comparison.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/remote-home/pxl/R2R/output/replay_full_prefill/router0_top1diff_rule_breakdown",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "router0_top1diff_rule_breakdown_detail.csv")

    summary_stats = {}
    total = 0

    with open(args.input, "r", encoding="utf-8") as f_in, open(
        detail_path, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        if "category" not in fieldnames:
            fieldnames = fieldnames + ["category", "category_reason"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                router_val = int(row.get("router_decision_replayed", 0))
            except ValueError:
                continue
            top1_same = row.get("top1_same", "")
            top1_same_bool = str(top1_same).strip().lower() in {"true", "1"}

            if router_val != 0 or top1_same_bool:
                continue

            category, reason = classify_pair(row.get("slm_top1_str"), row.get("llm_top1_str"))
            row["category"] = category
            row["category_reason"] = reason
            writer.writerow(row)

            total += 1
            stats = summary_stats.setdefault(
                category,
                {"count": 0, "js": [], "entropy": []},
            )
            stats["count"] += 1
            try:
                stats["js"].append(float(row.get("js", "nan")))
            except ValueError:
                pass
            try:
                stats["entropy"].append(float(row.get("slm_entropy", "nan")))
            except ValueError:
                pass

    summary_rows = []
    for category, stats in summary_stats.items():
        js_vals = [v for v in stats["js"] if v == v]
        ent_vals = [v for v in stats["entropy"] if v == v]
        js_mean = sum(js_vals) / len(js_vals) if js_vals else float("nan")
        ent_mean = sum(ent_vals) / len(ent_vals) if ent_vals else float("nan")
        js_median = statistics.median(js_vals) if js_vals else float("nan")
        ent_median = statistics.median(ent_vals) if ent_vals else float("nan")
        summary_rows.append(
            {
                "category": category,
                "count": stats["count"],
                "ratio": stats["count"] / total if total else 0.0,
                "js_mean": js_mean,
                "js_median": js_median,
                "entropy_mean": ent_mean,
                "entropy_median": ent_median,
            }
        )

    summary_rows.sort(key=lambda x: x["count"], reverse=True)

    summary_csv = os.path.join(args.output_dir, "router0_top1diff_rule_breakdown_summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f_sum:
        writer = csv.DictWriter(
            f_sum,
            fieldnames=[
                "category",
                "count",
                "ratio",
                "js_mean",
                "js_median",
                "entropy_mean",
                "entropy_median",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = os.path.join(args.output_dir, "router0_top1diff_rule_breakdown_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {"total": int(total), "categories": summary_rows},
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
import glob
import os
from collections import defaultdict


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def sort_key(problem_id: str):
    text = str(problem_id)
    return (0, int(text)) if text.isdigit() else (1, text)


def get_problem_statuses(temp_csv_dir: str):
    statuses = defaultdict(lambda: {"has_true": False, "has_false": False})

    for csv_path in glob.glob(os.path.join(temp_csv_dir, "*.csv")):
        try:
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue

        if not rows:
            continue

        row = rows[0]
        problem_id = str(
            row.get("problem_id")
            or os.path.basename(csv_path).split("_run_")[0]
        )
        has_extracted_answer = parse_bool(row.get("has_extracted_answer", ""))

        if has_extracted_answer:
            statuses[problem_id]["has_true"] = True
        else:
            statuses[problem_id]["has_false"] = True

    return statuses


def main():
    parser = argparse.ArgumentParser(
        description="Find problem IDs without any extracted answers in temp_csv results."
    )
    parser.add_argument("--output_dir", required=True, help="Evaluation output directory")
    args = parser.parse_args()

    temp_csv_dir = os.path.join(args.output_dir, "temp_csv")
    if not os.path.exists(temp_csv_dir):
        print("", end="")
        return

    statuses = get_problem_statuses(temp_csv_dir)
    missing_problem_ids = sorted(
        [
            problem_id
            for problem_id, status in statuses.items()
            if status["has_false"] and not status["has_true"]
        ],
        key=sort_key,
    )

    print(",".join(missing_problem_ids), end="")


if __name__ == "__main__":
    main()

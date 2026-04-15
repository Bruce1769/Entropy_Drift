#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


RUN_FILE_PATTERN = re.compile(r"^(?P<problem_id>.+)_run_(?P<run_id>\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate shard_* online eval outputs into a single experiment directory."
    )
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def find_experiment_dirs(base_output_dir: Path) -> List[Path]:
    shard_dirs = sorted(
        path
        for path in base_output_dir.iterdir()
        if path.is_dir() and path.name.startswith("shard_")
    )
    if shard_dirs:
        return shard_dirs

    if (base_output_dir / "args.json").is_file() and (base_output_dir / "model_config.json").is_file():
        return [base_output_dir]

    raise FileNotFoundError(
        f"No args.json/model_config.json or shard_* directories found in {base_output_dir}"
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_combined_results_or_temp_csv(experiment_dir: Path) -> pd.DataFrame:
    combined_path = experiment_dir / "combined_results.csv"
    if combined_path.is_file():
        return pd.read_csv(combined_path)

    temp_csv_dir = experiment_dir / "temp_csv"
    if not temp_csv_dir.is_dir():
        raise FileNotFoundError(
            f"Neither combined_results.csv nor temp_csv/ found in {experiment_dir}"
        )

    records: List[Tuple[str, int, Path]] = []
    for csv_path in sorted(temp_csv_dir.glob("*.csv")):
        match = RUN_FILE_PATTERN.match(csv_path.name)
        if not match:
            continue
        records.append(
            (
                match.group("problem_id"),
                int(match.group("run_id")),
                csv_path,
            )
        )

    if not records:
        raise FileNotFoundError(f"No temp CSV files found under {temp_csv_dir}")

    dfs = []
    for problem_id, run_id, csv_path in records:
        df = pd.read_csv(csv_path)
        df["__problem_id_from_filename"] = problem_id
        df["__run_id"] = run_id
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    if "problem_id" in merged.columns:
        merged["problem_id"] = merged["problem_id"].astype(str)
    else:
        merged["problem_id"] = merged["__problem_id_from_filename"].astype(str)

    merged = (
        merged.sort_values(["problem_id", "__run_id"], ascending=[True, False])
        .drop_duplicates(subset=["problem_id"], keep="first")
        .reset_index(drop=True)
    )
    return merged


def aggregate_experiment_dirs(
    base_output_dir: Path,
) -> tuple[dict, dict, pd.DataFrame, List[Path]]:
    experiment_dirs = find_experiment_dirs(base_output_dir)
    first_args = load_json(experiment_dirs[0] / "args.json")
    first_model_config = load_json(experiment_dirs[0] / "model_config.json")

    combined_dfs = []
    for experiment_dir in experiment_dirs:
        args_json = load_json(experiment_dir / "args.json")
        model_config = load_json(experiment_dir / "model_config.json")
        if args_json.get("dataset") != first_args.get("dataset"):
            raise ValueError(
                f"Mismatched dataset across shards: {experiment_dir} has {args_json.get('dataset')}, "
                f"expected {first_args.get('dataset')}"
            )
        if model_config.get("quick", {}).get("model_path") != first_model_config.get("quick", {}).get("model_path"):
            raise ValueError(f"Mismatched quick model across shards: {experiment_dir}")
        if model_config.get("reference", {}).get("model_path") != first_model_config.get("reference", {}).get("model_path"):
            raise ValueError(f"Mismatched reference model across shards: {experiment_dir}")

        df = load_combined_results_or_temp_csv(experiment_dir).copy()
        df["__experiment_dir"] = str(experiment_dir)
        df["__experiment_name"] = experiment_dir.name
        combined_dfs.append(df)

    merged = pd.concat(combined_dfs, ignore_index=True)
    if "problem_id" in merged.columns:
        merged["problem_id"] = merged["problem_id"].astype(str)
        merged = merged.sort_values("problem_id").reset_index(drop=True)

    return first_args, first_model_config, merged, experiment_dirs


def main() -> None:
    args = parse_args()
    base_output_dir = Path(args.base_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    args_json, model_config, merged, experiment_dirs = aggregate_experiment_dirs(base_output_dir)

    aggregated_args = dict(args_json)
    aggregated_args["output_dir"] = str(output_dir)
    aggregated_args["aggregated_from"] = [str(path) for path in experiment_dirs]

    (output_dir / "args.json").write_text(
        json.dumps(aggregated_args, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    merged.to_csv(output_dir / "combined_results.csv", index=False)

    summary = {
        "base_output_dir": str(base_output_dir),
        "output_dir": str(output_dir),
        "num_experiment_dirs": len(experiment_dirs),
        "experiment_dirs": [str(path) for path in experiment_dirs],
        "num_rows": int(len(merged)),
        "num_problem_ids": int(merged["problem_id"].nunique()) if "problem_id" in merged.columns else None,
    }
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Aggregated {len(experiment_dirs)} experiment directories into {output_dir}")
    print(f"Wrote combined results to {output_dir / 'combined_results.csv'}")


if __name__ == "__main__":
    main()

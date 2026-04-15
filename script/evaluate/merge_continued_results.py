import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

from script.evaluate.hf_dataset_sglang import extract_results_from_temp_csvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a continuation run into a fresh copy of the original output directory."
    )
    parser.add_argument("--source-output-dir", required=True, help="Original 8192 run directory.")
    parser.add_argument("--continuation-output-dir", required=True, help="Continuation run directory.")
    parser.add_argument("--merged-output-dir", required=True, help="New merged directory to create.")
    parser.add_argument(
        "--allow-existing-empty-dir",
        action="store_true",
        help="Allow merged-output-dir to already exist if it is empty.",
    )
    return parser.parse_args()


def ensure_dir_exists(path: str, label: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def copy_tree(src: str, dst: str, allow_existing_empty_dir: bool = False) -> None:
    if os.path.exists(dst):
        if not allow_existing_empty_dir:
            raise FileExistsError(f"merged output dir already exists: {dst}")
        if os.listdir(dst):
            raise FileExistsError(f"merged output dir exists and is not empty: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_matching_files(src_dir: str, dst_dir: str) -> List[str]:
    copied: List[str] = []
    if not os.path.isdir(src_dir):
        return copied
    os.makedirs(dst_dir, exist_ok=True)
    for name in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            copied.append(name)
    return copied


def remove_matching_entries(base_dir: str, prefixes: List[str], suffix: str = "") -> List[str]:
    removed: List[str] = []
    if not os.path.isdir(base_dir):
        return removed
    for name in sorted(os.listdir(base_dir)):
        if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        if suffix and not name.endswith(suffix):
            continue
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        removed.append(name)
    return removed


def build_merged_results_csv(merged_output_dir: str) -> str:
    extract_results_from_temp_csvs(merged_output_dir, use_job_dirs=False)
    combined_path = os.path.join(merged_output_dir, "combined_results.csv")
    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"combined_results.csv was not generated in {merged_output_dir}")

    df = pd.read_csv(combined_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(merged_output_dir, f"results_merged_gpu-1_thread-1_{timestamp}.csv")
    df.to_csv(results_path, index=False)
    return results_path


def write_manifest(merged_output_dir: str, manifest: Dict) -> str:
    manifest_path = os.path.join(merged_output_dir, "merge_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main() -> None:
    args = parse_args()

    source_output_dir = os.path.abspath(args.source_output_dir)
    continuation_output_dir = os.path.abspath(args.continuation_output_dir)
    merged_output_dir = os.path.abspath(args.merged_output_dir)

    ensure_dir_exists(source_output_dir, "source output dir")
    ensure_dir_exists(continuation_output_dir, "continuation output dir")
    ensure_parent(merged_output_dir)

    copy_tree(
        src=source_output_dir,
        dst=merged_output_dir,
        allow_existing_empty_dir=args.allow_existing_empty_dir,
    )

    copied_temp = copy_matching_files(
        os.path.join(continuation_output_dir, "temp"),
        os.path.join(merged_output_dir, "temp"),
    )
    copied_temp_csv = copy_matching_files(
        os.path.join(continuation_output_dir, "temp_csv"),
        os.path.join(merged_output_dir, "temp_csv"),
    )
    copied_token_traces = copy_matching_files(
        os.path.join(continuation_output_dir, "token_traces"),
        os.path.join(merged_output_dir, "token_traces"),
    )

    removed_results = remove_matching_entries(
        merged_output_dir,
        prefixes=["results_"],
        suffix=".csv",
    )
    removed_outputs = remove_matching_entries(
        merged_output_dir,
        prefixes=["outputs_"],
    )

    merged_results_path = build_merged_results_csv(merged_output_dir)

    stats_path = None
    combined_path = os.path.join(merged_output_dir, "combined_results.csv")
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
        stats = {
            "row_count": int(len(df)),
        }
        for col in ["has_extracted_answer", "is_correct"]:
            if col in df.columns:
                stats[f"avg_{col}"] = float(df[col].mean() * 100)
        for col in [
            "quick_model_percentage",
            "reference_model_percentage",
            "reference_eval_ratio",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reference_eval_count",
        ]:
            if col in df.columns:
                stats[f"avg_{col}"] = float(df[col].mean())
        stats_path = os.path.join(merged_output_dir, "stats.csv")
        pd.DataFrame(list(stats.items()), columns=["metric_name", "value"]).to_csv(
            stats_path, index=False
        )

    manifest = {
        "source_output_dir": source_output_dir,
        "continuation_output_dir": continuation_output_dir,
        "merged_output_dir": merged_output_dir,
        "copied_temp_files": copied_temp,
        "copied_temp_csv_files": copied_temp_csv,
        "copied_token_trace_files": copied_token_traces,
        "removed_results_files": removed_results,
        "removed_outputs_dirs": removed_outputs,
        "merged_results_csv": merged_results_path,
        "combined_results_csv": combined_path if os.path.exists(combined_path) else None,
        "stats_csv": stats_path,
    }
    manifest_path = write_manifest(merged_output_dir, manifest)

    print(f"Merged directory created at: {merged_output_dir}")
    print(f"Overlaid {len(copied_temp_csv)} temp_csv files from continuation run")
    print(f"Rebuilt combined results: {combined_path}")
    print(f"Wrote merged results file: {merged_results_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

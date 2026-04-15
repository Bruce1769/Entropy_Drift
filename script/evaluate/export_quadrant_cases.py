#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from collections import defaultdict

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def load_experiment_metadata(experiment_dir: str):
    args_path = os.path.join(experiment_dir, "args.json")
    model_config_path = os.path.join(experiment_dir, "model_config.json")
    combined_results_path = os.path.join(experiment_dir, "combined_results.csv")

    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"args.json not found: {args_path}")
    if not os.path.isfile(model_config_path):
        raise FileNotFoundError(f"model_config.json not found: {model_config_path}")
    if not os.path.isfile(combined_results_path):
        raise FileNotFoundError(f"combined_results.csv not found: {combined_results_path}")

    with open(args_path, "r") as f:
        args = json.load(f)
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    return args, model_config, combined_results_path


def resolve_experiment_dir(comparison_csv: str, experiment_dir: str | None):
    if experiment_dir:
        return experiment_dir
    base = comparison_csv.replace("_comparison.csv", "")
    summary_path = f"{base}_summary.json"
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        exp = summary.get("experiment_dir")
        if exp:
            return exp
    raise ValueError("experiment_dir not provided and could not be inferred from summary json")


def _dataset_cache_root():
    return os.environ.get("HF_DATASETS_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "datasets"
    )


def _dataset_slug(dataset_path: str):
    return dataset_path.replace("/", "___")


def _load_from_cache_arrow(dataset_path: str):
    cache_root = _dataset_cache_root()
    slug = _dataset_slug(dataset_path)
    base = os.path.join(cache_root, slug, "default", "0.0.0")
    if not os.path.isdir(base):
        return None
    for root, _, files in os.walk(base):
        for name in files:
            if name.endswith(".arrow"):
                return Dataset.from_file(os.path.join(root, name))
    return None


def load_questions(dataset_path: str, dataset_config: str | None, id_field: str, question_field: str):
    def _load(split: str):
        if dataset_config:
            return load_dataset(
                dataset_path,
                dataset_config,
                split=split,
                download_mode="reuse_dataset_if_exists",
                local_files_only=True,
            )
        return load_dataset(
            dataset_path,
            split=split,
            download_mode="reuse_dataset_if_exists",
            local_files_only=True,
        )

    try:
        ds = _load("test")
    except Exception:
        try:
            ds = _load("train")
        except Exception:
            ds = _load_from_cache_arrow(dataset_path)
            if ds is None:
                raise RuntimeError(
                    f"Failed to load dataset {dataset_path} from cache. "
                    "Ensure it is cached locally."
                )

    question_by_id = {}
    for row in ds:
        question_by_id[str(row[id_field])] = str(row[question_field])
    return question_by_id


def load_full_outputs(combined_results_path: str):
    csv.field_size_limit(10 * 1024 * 1024)
    outputs = {}
    with open(combined_results_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            outputs[str(row["problem_id"])] = row.get("full_output", "")
    return outputs


def quadrant_key(router_decision: str, top1_same: str):
    router_val = str(router_decision).strip()
    top1_val = str(top1_same).strip()
    router_is_1 = router_val in {"1", "True", "true"}
    top1_is_same = top1_val in {"1", "True", "true"}
    if router_is_1 and top1_is_same:
        return "router1_top1same"
    if router_is_1 and not top1_is_same:
        return "router1_top1diff"
    if (not router_is_1) and top1_is_same:
        return "router0_top1same"
    return "router0_top1diff"


def reservoir_add(per_sample, sample_id, item, cap, rng):
    bucket = per_sample[sample_id]
    bucket["seen"] += 1
    seen = bucket["seen"]
    items = bucket["items"]
    if len(items) < cap:
        items.append(item)
        return
    j = rng.randint(0, seen - 1)
    if j < cap:
        items[j] = item


def round_robin_collect(per_sample, target, rng):
    sample_ids = list(per_sample.keys())
    rng.shuffle(sample_ids)
    for sid in sample_ids:
        rng.shuffle(per_sample[sid]["items"])
    out = []
    cursor = 0
    while len(out) < target and sample_ids:
        sid = sample_ids[cursor % len(sample_ids)]
        items = per_sample[sid]["items"]
        if items:
            out.append(items.pop())
        else:
            sample_ids.remove(sid)
            if not sample_ids:
                break
            cursor -= 1
        cursor += 1
    return out


def main():
    parser = argparse.ArgumentParser(description="Export representative cases for each router/top1 quadrant.")
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-quadrant", type=int, default=100)
    parser.add_argument("--per-sample-cap", type=int, default=6)
    parser.add_argument("--context-window", type=int, default=20)
    parser.add_argument("--question-preview-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    experiment_dir = resolve_experiment_dir(args.comparison_csv, args.experiment_dir)
    exp_args, model_config, combined_results_path = load_experiment_metadata(experiment_dir)

    dataset_cfg = exp_args.get("dataset_config_dict", {})
    dataset_path = dataset_cfg.get("path")
    if not dataset_path:
        raise ValueError("dataset_config_dict.path missing in args.json")
    dataset_config = exp_args.get("dataset_config", None)
    id_field = dataset_cfg.get("id_field", "problem_idx")
    question_field = dataset_cfg.get("question_field", "problem")

    question_by_id = load_questions(dataset_path, dataset_config, id_field, question_field)
    full_output_by_id = load_full_outputs(combined_results_path)

    quick_path = model_config["quick"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        quick_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    rng = random.Random(args.seed)
    per_quadrant = {
        "router1_top1same": defaultdict(lambda: {"seen": 0, "items": []}),
        "router1_top1diff": defaultdict(lambda: {"seen": 0, "items": []}),
        "router0_top1same": defaultdict(lambda: {"seen": 0, "items": []}),
        "router0_top1diff": defaultdict(lambda: {"seen": 0, "items": []}),
    }

    with open(args.comparison_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qkey = quadrant_key(row.get("router_decision_replayed"), row.get("top1_same"))
            sample_id = str(row.get("sample_id", "")).strip()
            reservoir_add(
                per_quadrant[qkey],
                sample_id,
                row,
                args.per_sample_cap,
                rng,
            )

    os.makedirs(args.output_dir, exist_ok=True)

    output_id_cache = {}

    def get_output_ids(sample_id: str):
        if sample_id in output_id_cache:
            return output_id_cache[sample_id]
        text = full_output_by_id.get(sample_id, "")
        output_ids = tokenizer.encode(text, add_special_tokens=False)
        output_id_cache[sample_id] = output_ids
        return output_ids

    for qkey, per_sample in per_quadrant.items():
        selected = round_robin_collect(per_sample, args.per_quadrant, rng)
        out_path = os.path.join(args.output_dir, f"{qkey}_cases.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample_id",
                    "position",
                    "js",
                    "slm_entropy",
                    "output_token_str",
                    "slm_top1_str",
                    "llm_top1_str",
                    "question_preview",
                    "context_text",
                ]
            )
            for row in selected:
                sample_id = str(row.get("sample_id", "")).strip()
                position = int(row.get("position", "0"))
                output_ids = get_output_ids(sample_id)
                left = max(0, position - args.context_window)
                right = min(len(output_ids), position + args.context_window + 1)
                context_text = tokenizer.decode(output_ids[left:right], skip_special_tokens=False)
                question = question_by_id.get(sample_id, "")
                question_preview = question.replace("\n", " ").strip()[: args.question_preview_len]
                writer.writerow(
                    [
                        sample_id,
                        position,
                        row.get("js"),
                        row.get("slm_entropy"),
                        row.get("output_token_str"),
                        row.get("slm_top1_str"),
                        row.get("llm_top1_str"),
                        question_preview,
                        context_text,
                    ]
                )


if __name__ == "__main__":
    main()

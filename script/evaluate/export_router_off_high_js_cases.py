import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import pandas as pd
from datasets import Dataset, DownloadConfig, load_dataset


def ensure_sklearn_stub() -> None:
    if "sklearn" in sys.modules:
        return

    sklearn_module = types.ModuleType("sklearn")
    metrics_module = types.ModuleType("sklearn.metrics")

    def roc_curve(*args, **kwargs):
        raise RuntimeError("sklearn.metrics.roc_curve is unavailable")

    sklearn_module.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    metrics_module.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
    sklearn_module.metrics = metrics_module
    metrics_module.roc_curve = roc_curve
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.metrics"] = metrics_module


def import_tokenizer():
    ensure_sklearn_stub()
    from transformers import AutoTokenizer

    return AutoTokenizer


def resolve_local_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.exists():
        return str(path)

    if "/" not in model_path:
        return model_path

    org, repo = model_path.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{repo}"
    snapshots_dir = cache_root / "snapshots"
    refs_main = cache_root / "refs" / "main"

    snapshot_path = None
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        candidate = snapshots_dir / revision
        if candidate.exists():
            snapshot_path = candidate

    if snapshot_path is None and snapshots_dir.exists():
        snapshots = sorted(
            [snapshot for snapshot in snapshots_dir.iterdir() if snapshot.is_dir()],
            key=lambda snapshot: snapshot.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            snapshot_path = snapshots[0]

    if snapshot_path and snapshot_path.exists():
        return str(snapshot_path)

    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export case-study rows where router=0 but JS divergence is high."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--js-threshold", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--max-per-sample", type=int, default=20)
    parser.add_argument("--context-window", type=int, default=24)
    return parser.parse_args()


def load_processed_dataset(args_json: dict) -> dict[str, dict]:
    dataset_config = args_json["dataset_config_dict"]
    dataset_path = args_json["dataset_path"]
    dataset_config_name = args_json.get("dataset_config")
    offline_download_config = DownloadConfig(local_files_only=True)

    def try_load_from_arrow_cache():
        if "/" not in dataset_path:
            return None
        cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
        org, repo = dataset_path.split("/", 1)
        dataset_cache = cache_root / f"{org}___{repo}"
        if not dataset_cache.exists():
            return None
        arrow_files = sorted(dataset_cache.glob("default/*/*/*-train.arrow"))
        if not arrow_files:
            return None
        return Dataset.from_file(str(arrow_files[0]))

    dataset = try_load_from_arrow_cache()
    if dataset is None:
        if dataset_config_name:
            dataset = load_dataset(
                dataset_path,
                dataset_config_name,
                split="train",
                download_config=offline_download_config,
            )
        else:
            dataset = load_dataset(
                dataset_path,
                split="train",
                download_config=offline_download_config,
            )

    id_field = dataset_config.get("id_field", "ID")
    question_field = dataset_config.get("question_field", "Problem")
    prompt_template = dataset_config.get("prompt_template") or "{question}"

    mapping = {}
    for idx, item in enumerate(dataset):
        item_id = str(item.get(id_field, idx + 1))
        question = item[question_field]
        mapping[item_id] = {
            "question": question,
            "prompt": prompt_template.format(question=question),
        }
    return mapping


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    args_json = json.loads((experiment_dir / "args.json").read_text())
    model_config = json.loads((experiment_dir / "model_config.json").read_text())
    combined = pd.read_csv(experiment_dir / "combined_results.csv")
    comparison = pd.read_csv(args.comparison)

    comparison["sample_id"] = comparison["sample_id"].astype(str)
    combined["problem_id"] = combined["problem_id"].astype(str)

    filtered = comparison[
        (comparison["router_decision_replayed"] == 0) & (comparison["js"] >= args.js_threshold)
    ].copy()
    filtered = filtered.sort_values(["js", "slm_entropy"], ascending=[False, False])
    filtered["rank_within_sample"] = filtered.groupby("sample_id").cumcount() + 1
    filtered = filtered[filtered["rank_within_sample"] <= args.max_per_sample].head(args.top_n)

    if filtered.empty:
        pd.DataFrame().to_csv(args.output_csv, index=False)
        print(f"No rows matched; wrote empty csv to {args.output_csv}")
        return

    dataset_mapping = load_processed_dataset(args_json)
    AutoTokenizer = import_tokenizer()
    tokenizer_path = resolve_local_model_path(model_config["quick"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)

    full_output_map = {
        str(row["problem_id"]): str(row["full_output"])
        for _, row in combined.iterrows()
    }

    case_rows = []
    for _, row in filtered.iterrows():
        sample_id = str(row["sample_id"])
        full_output = full_output_map[sample_id]
        output_ids = tokenizer.encode(full_output, add_special_tokens=False)
        pos = int(row["position"])
        left = max(0, pos - args.context_window)
        right = min(len(output_ids), pos + args.context_window + 1)
        context_ids = output_ids[left:right]
        context_text = tokenizer.decode(context_ids)
        focus_token = tokenizer.decode([output_ids[pos]]) if pos < len(output_ids) else ""

        q = dataset_mapping.get(sample_id, {})
        question = q.get("question", "")
        question_preview = question[:400]

        case_rows.append(
            {
                "sample_id": sample_id,
                "position": pos,
                "js": row["js"],
                "slm_entropy": row["slm_entropy"],
                "top1_same": row["top1_same"],
                "router_decision_replayed": row["router_decision_replayed"],
                "output_token_id": row["output_token_id"],
                "output_token_str": row["output_token_str"],
                "focus_token_decoded": focus_token,
                "slm_top1_id": row["slm_top1_id"],
                "slm_top1_str": row["slm_top1_str"],
                "llm_top1_id": row["llm_top1_id"],
                "llm_top1_str": row["llm_top1_str"],
                "question_preview": question_preview,
                "context_left_pos": left,
                "context_right_pos_exclusive": right,
                "context_text": context_text,
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(case_rows).to_csv(output_path, index=False)
    print(f"Wrote {len(case_rows)} case rows to {output_path}")


if __name__ == "__main__":
    main()

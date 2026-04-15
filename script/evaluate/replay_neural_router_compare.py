import argparse
import importlib.machinery
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import types

import pandas as pd
import torch
from datasets import Dataset, DownloadConfig, load_dataset

from r2r.utils.dataclass import ModelOutputs
from r2r.utils.metrics import compute_entropy, compute_js_divergence
from r2r.utils.switching import create_switching_strategy


def ensure_sklearn_stub() -> None:
    if "sklearn" in sys.modules:
        return

    sklearn_module = types.ModuleType("sklearn")
    metrics_module = types.ModuleType("sklearn.metrics")

    def roc_curve(*args, **kwargs):
        raise RuntimeError("sklearn.metrics.roc_curve is unavailable in replay_neural_router_compare.py")

    sklearn_module.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    metrics_module.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
    sklearn_module.metrics = metrics_module
    metrics_module.roc_curve = roc_curve

    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.metrics"] = metrics_module


def import_transformers():
    ensure_sklearn_stub()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer


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
        description="Replay a neural-router experiment on final-text prefixes and export per-position comparisons."
    )
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory containing args.json and combined_results.csv.")
    parser.add_argument("--sample-id", default=None, help="Optional single sample/problem id to replay.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional number of samples to replay.")
    parser.add_argument("--max-positions", type=int, default=None, help="Optional cap on replayed output positions per sample.")
    parser.add_argument("--quick-device", default="cuda:0", help="Device for quick model.")
    parser.add_argument("--reference-device", default="cuda:1", help="Device for reference model.")
    parser.add_argument("--router-device", default="cuda:2", help="Device for router.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Model dtype.")
    parser.add_argument("--decode-chunk-size", type=int, default=256, help="Chunk size for teacher-forced decode over final text.")
    parser.add_argument("--router-batch-size", type=int, default=256, help="Batch size for batched router inference.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix for generated CSV/JSON.")
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_experiment_metadata(experiment_dir: Path) -> tuple[dict, dict, pd.DataFrame]:
    args = json.loads((experiment_dir / "args.json").read_text())
    model_config = json.loads((experiment_dir / "model_config.json").read_text())
    combined = pd.read_csv(experiment_dir / "combined_results.csv")
    return args, model_config, combined


def load_processed_dataset(args: dict, experiment_dir: Path) -> Dict[str, dict]:
    dataset_name = args["dataset"]
    dataset_config = args["dataset_config_dict"]
    dataset_path = args["dataset_path"]
    dataset_config_name = args.get("dataset_config")
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

        dataset = Dataset.from_file(str(arrow_files[0]))
        print(f"[replay] loaded dataset from arrow cache: {arrow_files[0]}", flush=True)
        return dataset

    dataset = try_load_from_arrow_cache()
    if dataset is None:
        try:
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
            print(f"[replay] loaded dataset from local cache: {dataset_path}", flush=True)
        except Exception as offline_error:
            print(
                f"[replay] local dataset cache unavailable, falling back to normal load: {offline_error}",
                flush=True,
            )
            if dataset_config_name:
                dataset = load_dataset(dataset_path, dataset_config_name, split="train")
            else:
                dataset = load_dataset(dataset_path, split="train")

    id_field = dataset_config.get("id_field", "ID")
    question_field = dataset_config.get("question_field", "Problem")
    answer_field = dataset_config.get("answer_field", "Answer")
    prompt_template = dataset_config.get("prompt_template") or "{question}"

    mapping: Dict[str, dict] = {}
    for idx, item in enumerate(dataset):
        item_id = item.get(id_field, idx + 1)
        prompt = prompt_template.format(question=item[question_field])
        mapping[str(item_id)] = {
            "ID": str(item_id),
            "Problem": item[question_field],
            "Answer": item[answer_field],
            "FormattedProblem": prompt,
        }
    return mapping


def select_rows(combined: pd.DataFrame, sample_id: Optional[str], max_samples: Optional[int]) -> pd.DataFrame:
    df = combined.copy()
    df["problem_id"] = df["problem_id"].astype(str)
    if sample_id is not None:
        df = df[df["problem_id"] == str(sample_id)]
    if max_samples is not None:
        df = df.head(max_samples)
    if df.empty:
        raise ValueError("No matching samples found in combined_results.csv")
    return df


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def build_router(args_json: dict, model_config: dict, device: str):
    router_config = model_config["router"]
    router_path = resolve_local_model_path(router_config["router_path"])
    router_path_obj = Path(router_path)
    if router_path_obj.is_dir():
        default_router = router_path_obj / "default_router.pt"
        if default_router.exists():
            router_path = str(default_router)
    override_init_args = dict(router_config.get("override_init_args", {}))
    if "pretrained_model_name" in override_init_args:
        override_init_args["pretrained_model_name"] = resolve_local_model_path(
            override_init_args["pretrained_model_name"]
        )
    strategy_kwargs = {
        "model_path": router_path,
        "threshold": args_json.get("threshold"),
        "device": device,
        "use_cuda_graph": False,
        "override_init_args": override_init_args,
    }
    return create_switching_strategy("neural", **strategy_kwargs)


def as_device_map(device: str) -> dict:
    return {"": device}


def replay_single_sample(
    row: pd.Series,
    prompt: str,
    quick_model,
    reference_model,
    tokenizer,
    router,
    max_positions: Optional[int],
    quick_device: str,
    reference_device: str,
    decode_chunk_size: int,
    router_batch_size: int,
) -> tuple[list[dict], list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    output_ids = tokenizer.encode(str(row["full_output"]), add_special_tokens=False)
    if max_positions is not None:
        output_ids = output_ids[:max_positions]
    if not output_ids:
        return [], []

    output_len = len(output_ids)
    with torch.no_grad():
        quick_out = quick_model(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=quick_device),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        reference_out = reference_model(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=reference_device),
            use_cache=True,
            return_dict=True,
        )

    quick_past = quick_out.past_key_values
    reference_past = reference_out.past_key_values
    logits_chunks = [quick_out.logits[:, -1:, :]]
    reference_logits_chunks = [reference_out.logits[:, -1:, :].to(device=quick_device)]
    hidden_chunks = [quick_out.hidden_states[-1][:, -1:, :]]

    start = 0
    while start < output_len:
        chunk_ids = output_ids[start : start + decode_chunk_size]
        chunk_len = len(chunk_ids)
        with torch.no_grad():
            quick_out = quick_model(
                input_ids=torch.tensor([chunk_ids], dtype=torch.long, device=quick_device),
                past_key_values=quick_past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            reference_out = reference_model(
                input_ids=torch.tensor([chunk_ids], dtype=torch.long, device=reference_device),
                past_key_values=reference_past,
                use_cache=True,
                return_dict=True,
            )

        quick_past = quick_out.past_key_values
        reference_past = reference_out.past_key_values

        valid_len = min(chunk_len, output_len - 1 - start)
        if valid_len > 0:
            logits_chunks.append(quick_out.logits[:, :valid_len, :])
            reference_logits_chunks.append(reference_out.logits[:, :valid_len, :].to(device=quick_device))
            hidden_chunks.append(quick_out.hidden_states[-1][:, :valid_len, :])
        start += chunk_len

    quick_logits_all = torch.cat(logits_chunks, dim=1)[0]
    reference_logits_all = torch.cat(reference_logits_chunks, dim=1)[0]
    quick_hidden_all = torch.cat(hidden_chunks, dim=1)[0]
    slm_top1_ids = torch.argmax(quick_logits_all, dim=-1)
    llm_top1_ids = torch.argmax(reference_logits_all, dim=-1)

    rows: list[dict] = []
    route_positions: list[int] = []
    sample_id = str(row["problem_id"])

    route_decisions = []
    for start in range(0, output_len, router_batch_size):
        end = min(start + router_batch_size, output_len)
        batch_logits = quick_logits_all[start:end]
        batch_hidden = quick_hidden_all[start:end]
        batch_top1 = slm_top1_ids[start:end]
        model_outputs = ModelOutputs(
            logits=batch_logits[:, None, :],
            hidden_states=[batch_hidden[:, None, :]],
            token=batch_top1[:, None],
        )
        batch_routes = router.route(model_outputs).detach().cpu().tolist()
        route_decisions.extend(int(x) for x in batch_routes)

    for position, true_token_id in enumerate(output_ids):
        slm_top1_id = int(slm_top1_ids[position].item())
        llm_top1_id = int(llm_top1_ids[position].item())
        slm_entropy = float(compute_entropy(quick_logits_all[position]))
        js = float(
            compute_js_divergence(
                quick_logits_all[position].to(dtype=torch.float32, device="cpu"),
                reference_logits_all[position].to(dtype=torch.float32, device="cpu"),
            )
        )
        route_decision = route_decisions[position]
        if route_decision == 1:
            route_positions.append(position)

        rows.append(
            {
                "sample_id": sample_id,
                "position": position,
                "output_token_id": int(true_token_id),
                "output_token_str": decode_token(tokenizer, int(true_token_id)),
                "slm_top1_id": slm_top1_id,
                "slm_top1_str": decode_token(tokenizer, slm_top1_id),
                "llm_top1_id": llm_top1_id,
                "llm_top1_str": decode_token(tokenizer, llm_top1_id),
                "top1_same": slm_top1_id == llm_top1_id,
                "slm_entropy": slm_entropy,
                "js": js,
                "router_decision_replayed": route_decision,
            }
        )

    return rows, route_positions


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[replay] loading experiment metadata from {experiment_dir}", flush=True)
    args_json, model_config, combined = load_experiment_metadata(experiment_dir)
    print(f"[replay] loading dataset mapping for {args_json['dataset']}", flush=True)
    dataset_mapping = load_processed_dataset(args_json, experiment_dir)
    selected_rows = select_rows(combined, args.sample_id, args.max_samples)
    print(f"[replay] selected {len(selected_rows)} sample(s)", flush=True)

    dtype = get_torch_dtype(args.dtype)
    quick_path = resolve_local_model_path(model_config["quick"]["model_path"])
    reference_path = resolve_local_model_path(model_config["reference"]["model_path"])
    print(f"[replay] importing transformers", flush=True)
    AutoModelForCausalLM, AutoTokenizer = import_transformers()

    print(f"[replay] loading tokenizer from {quick_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(quick_path, trust_remote_code=True, local_files_only=True)
    print(f"[replay] loading quick model on {args.quick_device}", flush=True)
    quick_model = AutoModelForCausalLM.from_pretrained(
        quick_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map=as_device_map(args.quick_device),
    ).eval()
    print(f"[replay] loading reference model on {args.reference_device}", flush=True)
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map=as_device_map(args.reference_device),
    ).eval()
    print(f"[replay] loading neural router on {args.router_device}", flush=True)
    router = build_router(args_json, model_config, args.router_device)

    all_rows: list[dict] = []
    route_rows: list[dict] = []

    for _, row in selected_rows.iterrows():
        sample_id = str(row["problem_id"])
        if sample_id not in dataset_mapping:
            print(f"Skipping sample {sample_id}: prompt not found in dataset mapping")
            continue
        prompt = dataset_mapping[sample_id]["FormattedProblem"]
        print(f"[replay] sample {sample_id}: replaying", flush=True)
        per_pos_rows, route_positions = replay_single_sample(
            row=row,
            prompt=prompt,
            quick_model=quick_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            router=router,
            max_positions=args.max_positions,
            quick_device=args.quick_device,
            reference_device=args.reference_device,
            decode_chunk_size=args.decode_chunk_size,
            router_batch_size=args.router_batch_size,
        )
        all_rows.extend(per_pos_rows)
        route_rows.append(
            {
                "sample_id": sample_id,
                "route_count": len(route_positions),
                "route_positions": route_positions,
            }
        )
        print(f"Replayed sample {sample_id}: {len(per_pos_rows)} positions, {len(route_positions)} routed")

    pd.DataFrame(all_rows).to_csv(f"{output_prefix}_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                **row,
                "route_positions": json.dumps(row["route_positions"], ensure_ascii=False),
            }
            for row in route_rows
        ]
    ).to_csv(f"{output_prefix}_routes.csv", index=False)

    summary = {
        "experiment_dir": str(experiment_dir),
        "sample_count": len(route_rows),
        "position_rows": len(all_rows),
        "max_positions": args.max_positions,
    }
    with open(f"{output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote comparison rows to {output_prefix}_comparison.csv")
    print(f"Wrote route positions to {output_prefix}_routes.csv")
    print(f"Wrote summary to {output_prefix}_summary.json")


if __name__ == "__main__":
    main()

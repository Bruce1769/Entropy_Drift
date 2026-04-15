#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from aggregate_online_eval_shards import aggregate_experiment_dirs, find_experiment_dirs
from replay_neural_router_compare import (
    as_device_map,
    get_torch_dtype,
    import_transformers,
    load_experiment_metadata,
    load_processed_dataset,
    resolve_local_model_path,
    select_rows,
)
from r2r.utils.metrics import compute_entropy, compute_js_divergence


TRACE_FILE_PATTERN = re.compile(r"^(?P<problem_id>.+)_run_(?P<run_id>\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an entropy-threshold router experiment on final-text prefixes and export per-position comparisons."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--quick-device", default="cuda:0")
    parser.add_argument("--reference-device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--decode-chunk-size", type=int, default=256)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def load_experiment_bundle(experiment_dir: Path) -> tuple[dict, dict, pd.DataFrame, List[Path]]:
    experiment_dirs = find_experiment_dirs(experiment_dir)
    if len(experiment_dirs) == 1 and experiment_dirs[0] == experiment_dir:
        args_json, model_config, combined = load_experiment_metadata(experiment_dir)
        return args_json, model_config, combined, [experiment_dir]

    args_json, model_config, combined, experiment_dirs = aggregate_experiment_dirs(experiment_dir)
    return args_json, model_config, combined, experiment_dirs


def build_token_trace_index(experiment_dirs: List[Path]) -> Dict[str, Path]:
    best: Dict[str, tuple[int, Path]] = {}
    for experiment_dir in experiment_dirs:
        token_trace_dir = experiment_dir / "token_traces"
        if not token_trace_dir.is_dir():
            continue
        for csv_path in sorted(token_trace_dir.glob("*.csv")):
            match = TRACE_FILE_PATTERN.match(csv_path.name)
            if not match:
                continue
            problem_id = match.group("problem_id")
            run_id = int(match.group("run_id"))
            prev = best.get(problem_id)
            if prev is None or run_id > prev[0]:
                best[problem_id] = (run_id, csv_path)
    return {problem_id: csv_path for problem_id, (_, csv_path) in best.items()}


def decode_token(tokenizer, token_id: Optional[int]) -> Optional[str]:
    if token_id is None:
        return None
    return tokenizer.decode([int(token_id)])


def maybe_int(value):
    if value is None or pd.isna(value):
        return None
    return int(value)


def maybe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def maybe_str(value):
    if value is None or pd.isna(value):
        return None
    return str(value)


def quadrant_name(router_decision_replayed: int, top1_same: bool) -> str:
    if router_decision_replayed == 1 and top1_same:
        return "router1_top1same"
    if router_decision_replayed == 1 and not top1_same:
        return "router1_top1diff"
    if router_decision_replayed == 0 and top1_same:
        return "router0_top1same"
    return "router0_top1diff"


def replay_single_sample(
    row: pd.Series,
    prompt: str,
    quick_model,
    reference_model,
    tokenizer,
    max_positions: Optional[int],
    quick_device: str,
    reference_device: str,
    decode_chunk_size: int,
) -> tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    output_ids = tokenizer.encode(str(row["full_output"]), add_special_tokens=False)
    if max_positions is not None:
        output_ids = output_ids[:max_positions]
    if not output_ids:
        empty = torch.empty((0,), dtype=torch.long)
        empty_logits = torch.empty((0, 0), dtype=torch.float32)
        return [], empty_logits, empty_logits, empty, empty

    with torch.no_grad():
        quick_out = quick_model(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=quick_device),
            use_cache=True,
            return_dict=True,
        )
        reference_out = reference_model(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=reference_device),
            use_cache=True,
            return_dict=True,
        )

    quick_past = quick_out.past_key_values
    reference_past = reference_out.past_key_values
    quick_logits_chunks = [quick_out.logits[0, -1:, :].to(dtype=torch.float32, device="cpu")]
    reference_logits_chunks = [reference_out.logits[0, -1:, :].to(dtype=torch.float32, device="cpu")]

    start = 0
    output_len = len(output_ids)
    while start < output_len:
        chunk_ids = output_ids[start : start + decode_chunk_size]
        chunk_len = len(chunk_ids)
        with torch.no_grad():
            quick_out = quick_model(
                input_ids=torch.tensor([chunk_ids], dtype=torch.long, device=quick_device),
                past_key_values=quick_past,
                use_cache=True,
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
            quick_logits_chunks.append(
                quick_out.logits[0, :valid_len, :].to(dtype=torch.float32, device="cpu")
            )
            reference_logits_chunks.append(
                reference_out.logits[0, :valid_len, :].to(dtype=torch.float32, device="cpu")
            )
        start += chunk_len

    quick_logits_all = torch.cat(quick_logits_chunks, dim=0)
    reference_logits_all = torch.cat(reference_logits_chunks, dim=0)
    slm_top1_ids = torch.argmax(quick_logits_all, dim=-1)
    llm_top1_ids = torch.argmax(reference_logits_all, dim=-1)
    return output_ids, quick_logits_all, reference_logits_all, slm_top1_ids, llm_top1_ids


def replay_rows_for_sample(
    row: pd.Series,
    prompt: str,
    quick_model,
    reference_model,
    tokenizer,
    threshold: float,
    token_trace_path: Path,
    max_positions: Optional[int],
    quick_device: str,
    reference_device: str,
    decode_chunk_size: int,
) -> tuple[List[dict], dict]:
    online_trace = pd.read_csv(token_trace_path)
    online_trace["position"] = online_trace["position"].astype(int)
    online_by_position = {
        int(record["position"]): record
        for record in online_trace.to_dict(orient="records")
    }

    output_ids, quick_logits_all, reference_logits_all, slm_top1_ids, llm_top1_ids = replay_single_sample(
        row=row,
        prompt=prompt,
        quick_model=quick_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        max_positions=max_positions,
        quick_device=quick_device,
        reference_device=reference_device,
        decode_chunk_size=decode_chunk_size,
    )

    rows: List[dict] = []
    online_route_count = 0
    replay_route_count = 0
    agreement_count = 0

    for position, true_token_id in enumerate(output_ids):
        slm_top1_id = int(slm_top1_ids[position].item())
        llm_top1_id = int(llm_top1_ids[position].item())
        slm_entropy = float(compute_entropy(quick_logits_all[position]))
        js = float(
            compute_js_divergence(
                quick_logits_all[position],
                reference_logits_all[position],
            )
        )
        router_decision_replayed = int(slm_entropy >= threshold)
        top1_same = slm_top1_id == llm_top1_id
        quadrant = quadrant_name(router_decision_replayed, top1_same)

        online_record = online_by_position.get(position, {})
        online_router_decision = (
            maybe_int(online_record["router_decision"])
            if "router_decision" in online_record
            else None
        )
        if online_router_decision is not None:
            online_route_count += online_router_decision
            agreement_count += int(online_router_decision == router_decision_replayed)
        replay_route_count += router_decision_replayed

        output_token_id_online = (
            maybe_int(online_record["output_token_id"])
            if "output_token_id" in online_record
            else None
        )
        if output_token_id_online is None:
            output_token_id_online = int(true_token_id)
        output_token_str_online = (
            maybe_str(online_record["output_token_str"])
            if "output_token_str" in online_record
            else None
        )
        if output_token_str_online is None:
            output_token_str_online = decode_token(tokenizer, int(true_token_id))

        row_dict = {
            "sample_id": str(row["problem_id"]),
            "problem_id": str(row["problem_id"]),
            "position": position,
            "router_decision": online_router_decision,
            "router_score": maybe_float(online_record.get("router_score")),
            "router_threshold": maybe_float(online_record.get("router_threshold")),
            "router_name": maybe_str(online_record.get("router_name")),
            "quick_token_id": maybe_int(online_record.get("quick_token_id")),
            "quick_token_str": maybe_str(online_record.get("quick_token_str")),
            "output_token_id_online": output_token_id_online,
            "output_token_str_online": output_token_str_online,
            "source_model": maybe_str(online_record.get("source_model")),
            "output_token_id_replay": int(true_token_id),
            "output_token_str_replay": decode_token(tokenizer, int(true_token_id)),
            "slm_top1_id": slm_top1_id,
            "slm_top1_str": decode_token(tokenizer, slm_top1_id),
            "llm_top1_id": llm_top1_id,
            "llm_top1_str": decode_token(tokenizer, llm_top1_id),
            "top1_same": bool(top1_same),
            "slm_entropy": slm_entropy,
            "js": js,
            "router_decision_replayed": router_decision_replayed,
            "router_score_replayed": slm_entropy,
            "router_threshold_replayed": threshold,
            "quadrant": quadrant,
            "output_matches_slm_top1": output_token_id_online == slm_top1_id,
            "output_matches_llm_top1": output_token_id_online == llm_top1_id,
        }
        rows.append(row_dict)

    summary = {
        "sample_id": str(row["problem_id"]),
        "num_positions": len(output_ids),
        "online_route_count": online_route_count,
        "replay_route_count": replay_route_count,
        "online_replay_agreement_count": agreement_count,
        "token_trace_path": str(token_trace_path),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[replay-entropy] loading experiment metadata from {experiment_dir}", flush=True)
    args_json, model_config, combined, experiment_dirs = load_experiment_bundle(experiment_dir)
    if "problem_id" not in combined.columns:
        raise ValueError("combined results must contain a problem_id column")
    combined["problem_id"] = combined["problem_id"].astype(str)

    print(f"[replay-entropy] loading dataset mapping for {args_json['dataset']}", flush=True)
    dataset_mapping = load_processed_dataset(args_json, experiment_dir)
    selected_rows = select_rows(combined, args.sample_id, args.max_samples)
    print(f"[replay-entropy] selected {len(selected_rows)} sample(s)", flush=True)

    token_trace_index = build_token_trace_index(experiment_dirs)
    missing_trace_ids = [
        str(row["problem_id"]) for _, row in selected_rows.iterrows() if str(row["problem_id"]) not in token_trace_index
    ]
    if missing_trace_ids:
        raise FileNotFoundError(f"Missing token trace CSVs for sample ids: {missing_trace_ids}")

    dtype = get_torch_dtype(args.dtype)
    quick_path = resolve_local_model_path(model_config["quick"]["model_path"])
    reference_path = resolve_local_model_path(model_config["reference"]["model_path"])
    AutoModelForCausalLM, AutoTokenizer = import_transformers()

    print(f"[replay-entropy] loading tokenizer from {quick_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(quick_path, trust_remote_code=True, local_files_only=True)
    print(f"[replay-entropy] loading quick model on {args.quick_device}", flush=True)
    quick_model = AutoModelForCausalLM.from_pretrained(
        quick_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map=as_device_map(args.quick_device),
    ).eval()
    print(f"[replay-entropy] loading reference model on {args.reference_device}", flush=True)
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map=as_device_map(args.reference_device),
    ).eval()

    all_rows: List[dict] = []
    per_sample_summary: List[dict] = []
    for _, row in selected_rows.iterrows():
        sample_id = str(row["problem_id"])
        prompt = dataset_mapping[sample_id]["FormattedProblem"]
        sample_rows, sample_summary = replay_rows_for_sample(
            row=row,
            prompt=prompt,
            quick_model=quick_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            threshold=args.threshold,
            token_trace_path=token_trace_index[sample_id],
            max_positions=args.max_positions,
            quick_device=args.quick_device,
            reference_device=args.reference_device,
            decode_chunk_size=args.decode_chunk_size,
        )
        all_rows.extend(sample_rows)
        per_sample_summary.append(sample_summary)
        print(
            f"[replay-entropy] sample {sample_id}: positions={sample_summary['num_positions']}, "
            f"online_route_count={sample_summary['online_route_count']}, "
            f"replay_route_count={sample_summary['replay_route_count']}",
            flush=True,
        )

    merged_df = pd.DataFrame(all_rows)
    merged_csv_path = f"{output_prefix}_merged.csv"
    merged_df.to_csv(merged_csv_path, index=False)

    quadrant_counts = (
        merged_df.groupby("quadrant", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("quadrant")
        .reset_index(drop=True)
    )
    total_positions = max(int(len(merged_df)), 1)
    quadrant_counts["ratio"] = quadrant_counts["count"] / total_positions
    quadrant_counts_path = f"{output_prefix}_quadrant_counts.csv"
    quadrant_counts.to_csv(quadrant_counts_path, index=False)

    total_online_routes = sum(item["online_route_count"] for item in per_sample_summary)
    total_replay_routes = sum(item["replay_route_count"] for item in per_sample_summary)
    total_agreement = sum(item["online_replay_agreement_count"] for item in per_sample_summary)
    summary = {
        "experiment_dir": str(experiment_dir),
        "threshold": float(args.threshold),
        "num_samples": len(per_sample_summary),
        "num_positions": int(len(merged_df)),
        "online_route_count": int(total_online_routes),
        "replay_route_count": int(total_replay_routes),
        "online_route_ratio": float(total_online_routes / total_positions),
        "replay_route_ratio": float(total_replay_routes / total_positions),
        "online_replay_decision_agreement": float(total_agreement / total_positions),
        "merged_csv": merged_csv_path,
        "quadrant_counts_csv": quadrant_counts_path,
        "per_sample_summary": per_sample_summary,
    }
    summary_path = f"{output_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote merged comparison to {merged_csv_path}")
    print(f"Wrote quadrant counts to {quadrant_counts_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

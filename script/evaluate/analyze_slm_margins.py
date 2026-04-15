import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from replay_neural_router_compare import (
    get_torch_dtype,
    import_transformers,
    load_experiment_metadata,
    load_processed_dataset,
    resolve_local_model_path,
    select_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SLM top1/topk margins on replay outputs.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quick-device", default="cuda:5")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--decode-chunk-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def summarize(series: pd.Series) -> dict:
    series = series.dropna()
    if series.empty:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.9)),
        "p95": float(series.quantile(0.95)),
    }


def append_metrics_from_logits(metric_rows, sample_id: str, start_pos: int, logits: torch.Tensor, topk: int):
    probs = torch.softmax(logits, dim=-1)
    topk_probs, _ = torch.topk(probs, k=topk, dim=-1)
    topk_logits, _ = torch.topk(logits, k=topk, dim=-1)
    count = logits.shape[0]
    metric_rows.append(
        pd.DataFrame(
            {
                "sample_id": sample_id,
                "position": list(range(start_pos, start_pos + count)),
                "slm_top1_prob": topk_probs[:, 0].cpu().numpy(),
                "slm_top2_prob": topk_probs[:, 1].cpu().numpy(),
                "slm_prob_margin_top1_top2": (topk_probs[:, 0] - topk_probs[:, 1]).cpu().numpy(),
                "slm_logit_margin_top1_top2": (topk_logits[:, 0] - topk_logits[:, 1]).cpu().numpy(),
                "slm_top5_mass": topk_probs.sum(dim=-1).cpu().numpy(),
            }
        )
    )


def compute_slm_margin_rows_for_sample(
    row, prompt, tokenizer, quick_model, device: str, decode_chunk_size: int, topk: int
):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    output_ids = tokenizer.encode(str(row["full_output"]), add_special_tokens=False)
    if int(row["output_tokens"]) < len(output_ids):
        output_ids = output_ids[: int(row["output_tokens"])]
    output_len = len(output_ids)
    if output_len == 0:
        return []

    with torch.no_grad():
        quick_out = quick_model(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=device),
            use_cache=True,
            return_dict=True,
        )

    quick_past = quick_out.past_key_values
    metric_rows = []
    append_metrics_from_logits(metric_rows, str(row["problem_id"]), 0, quick_out.logits[0, -1:, :].to(torch.float32), topk)
    start = 0
    while start < output_len:
        chunk_ids = output_ids[start : start + decode_chunk_size]
        chunk_len = len(chunk_ids)
        with torch.no_grad():
            quick_out = quick_model(
                input_ids=torch.tensor([chunk_ids], dtype=torch.long, device=device),
                past_key_values=quick_past,
                use_cache=True,
                return_dict=True,
            )
        quick_past = quick_out.past_key_values
        valid_len = min(chunk_len, output_len - 1 - start)
        if valid_len > 0:
            append_metrics_from_logits(
                metric_rows,
                str(row["problem_id"]),
                start + 1,
                quick_out.logits[0, :valid_len, :].to(torch.float32),
                topk,
            )
        start += chunk_len

    return metric_rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args_json, model_config, combined = load_experiment_metadata(Path(args.experiment_dir))
    dataset_mapping = load_processed_dataset(args_json, Path(args.experiment_dir))
    selected_rows = select_rows(combined, None, None)
    comparison = pd.read_csv(args.comparison).copy()
    comparison["sample_id"] = comparison["sample_id"].astype(str)
    comparison["top1_same"] = comparison["top1_same"].astype(bool)
    comparison["quadrant"] = comparison.apply(
        lambda r: f"router{int(r['router_decision_replayed'])}_top1{'same' if r['top1_same'] else 'diff'}",
        axis=1,
    )

    AutoModelForCausalLM, AutoTokenizer = import_transformers()
    quick_path = resolve_local_model_path(model_config["quick"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(quick_path, trust_remote_code=True, local_files_only=True)
    quick_model = AutoModelForCausalLM.from_pretrained(
        quick_path,
        torch_dtype=get_torch_dtype(args.dtype),
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": args.quick_device},
    ).eval()

    metric_rows = []
    for _, row in selected_rows.iterrows():
        sample_id = str(row["problem_id"])
        prompt = dataset_mapping[sample_id]["FormattedProblem"]
        per_sample_metric_rows = compute_slm_margin_rows_for_sample(
            row, prompt, tokenizer, quick_model, args.quick_device, args.decode_chunk_size, args.topk
        )
        if not per_sample_metric_rows:
            continue
        metric_rows.extend(per_sample_metric_rows)

    margins = pd.concat(metric_rows, ignore_index=True)
    merged = comparison.merge(margins, on=["sample_id", "position"], how="left", validate="one_to_one")
    merged.to_csv(output_dir / "slm_margins_merged.csv", index=False)

    summary_rows = []
    for quadrant, sub in merged.groupby("quadrant"):
        summary_rows.append(
            {
                "quadrant": quadrant,
                "count": int(len(sub)),
                "prob_margin_mean": float(sub["slm_prob_margin_top1_top2"].mean()),
                "prob_margin_median": float(sub["slm_prob_margin_top1_top2"].median()),
                "logit_margin_mean": float(sub["slm_logit_margin_top1_top2"].mean()),
                "logit_margin_median": float(sub["slm_logit_margin_top1_top2"].median()),
                "top5_mass_mean": float(sub["slm_top5_mass"].mean()),
                "top5_mass_median": float(sub["slm_top5_mass"].median()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("quadrant")
    summary_df.to_csv(output_dir / "quadrant_margin_summary.csv", index=False)

    report = {
        quadrant: {
            "slm_prob_margin_top1_top2": summarize(sub["slm_prob_margin_top1_top2"]),
            "slm_logit_margin_top1_top2": summarize(sub["slm_logit_margin_top1_top2"]),
            "slm_top5_mass": summarize(sub["slm_top5_mass"]),
        }
        for quadrant, sub in merged.groupby("quadrant")
    }
    with open(output_dir / "quadrant_margin_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Wrote margin analysis to {output_dir}")


if __name__ == "__main__":
    main()

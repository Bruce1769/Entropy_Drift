import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from r2r.utils.cuda_host_compiler import ensure_cuda_host_compiler_for_jit

ensure_cuda_host_compiler_for_jit()

import pandas as pd
import torch
import yaml
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = "0"

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from r2r.evaluate.eval_utils import check_answer_correctness, get_answer_extractor
from r2r.models.sglang_patch.sl_disaggregation_system import SLDisaggregationSystem
from script.evaluate.hf_dataset_sglang import (
    DATASET_CONFIGS,
    combine_results,
    preprocess_dataset,
    resolve_local_model_path,
    save_results,
    save_token_trace_csv,
    write_to_csv,
    write_to_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue failed hf_dataset_sglang runs by appending prior full_output to the original prompt."
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--config-path", type=str, required=True, help="Hybrid config path.")
    parser.add_argument("--source-output-dir", type=str, required=True, help="Existing 8192 run output dir.")
    parser.add_argument("--output-dir", type=str, required=True, help="New output dir for continuation run.")
    parser.add_argument(
        "--filter-mode",
        type=str,
        default="missing_answer",
        choices=["missing_answer", "incorrect", "missing_answer_or_incorrect", "all", "problem_ids"],
        help="How to pick problems from the source output dir.",
    )
    parser.add_argument("--problem_ids", type=str, default=None, help="Comma-separated problem IDs to continue.")
    parser.add_argument("--skip_problem_ids", type=str, default=None, help="Comma-separated problem IDs to skip.")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--slm_tp_size", type=int, default=1)
    parser.add_argument("--llm_tp_size", type=int, default=1)
    parser.add_argument("--generator", type=str, default="sglang")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--previous_max_new_tokens", type=int, default=8192)
    parser.add_argument("--target_max_new_tokens", type=int, default=16384)
    parser.add_argument(
        "--continuation_max_new_tokens",
        type=int,
        default=None,
        help="Extra tokens to generate. Defaults to target_max_new_tokens - previous_max_new_tokens.",
    )
    parser.add_argument("--trace_reference_topk_k", type=int, default=64)
    parser.add_argument("--trace_reference_for_all_positions", action="store_true")
    parser.add_argument("--trace_logits_topk_k", type=int, default=0)
    parser.add_argument("--overlap_tp_schedule", action="store_true")
    parser.add_argument("--num_problems", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--reuse_reference_metrics",
        action="store_true",
        help="Approximate cumulative quick/reference percentages by combining old and new segments.",
    )

    args = parser.parse_args()

    if args.continuation_max_new_tokens is None:
        args.continuation_max_new_tokens = args.target_max_new_tokens - args.previous_max_new_tokens
    if args.continuation_max_new_tokens <= 0:
        parser.error("continuation_max_new_tokens must be positive.")

    with open(args.config_path, "r") as f:
        args.model_config = yaml.safe_load(f)

    quick = args.model_config["quick"]
    args.model_path = quick["model_path"]
    args.model_param = float(quick.get("param", 0))
    args.mem_fraction_static = float(quick.get("mem_fraction_static", 0.8))
    args.use_hybrid = True
    args.test_run_time = True
    args.is_record = True

    dataset_config = DATASET_CONFIGS[args.dataset]
    if args.dataset_path is None:
        args.dataset_path = dataset_config["path"]
    if args.dataset_config is None and "dataset_config" in dataset_config:
        args.dataset_config = dataset_config["dataset_config"]
    args.dataset_config_dict = dataset_config
    return args


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_problem_id_set(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def load_latest_source_rows(source_output_dir: str) -> Dict[str, Dict[str, Any]]:
    temp_csv_dir = os.path.join(source_output_dir, "temp_csv")
    if not os.path.isdir(temp_csv_dir):
        raise FileNotFoundError(f"temp_csv dir not found under {source_output_dir}")

    pattern = re.compile(r"^(?P<problem_id>.+)_run_(?P<run_number>\d+)\.csv$")
    latest: Dict[str, Tuple[int, Dict[str, Any]]] = {}

    for entry in os.listdir(temp_csv_dir):
        match = pattern.match(entry)
        if not match:
            continue
        problem_id = match.group("problem_id")
        run_number = int(match.group("run_number"))
        csv_path = os.path.join(temp_csv_dir, entry)
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        row = rows[-1]
        row["_source_temp_csv_path"] = csv_path
        row["_source_run_number"] = run_number
        prev = latest.get(problem_id)
        if prev is None or run_number > prev[0]:
            latest[problem_id] = (run_number, row)

    return {problem_id: row for problem_id, (_, row) in latest.items()}


def should_continue_row(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    if args.filter_mode == "all":
        return True
    if args.filter_mode == "problem_ids":
        include_ids = parse_problem_id_set(args.problem_ids)
        return include_ids is not None and str(row.get("problem_id")) in include_ids

    has_answer = parse_bool(row.get("has_extracted_answer"))
    is_correct = parse_bool(row.get("is_correct"))

    if args.filter_mode == "missing_answer":
        return not has_answer
    if args.filter_mode == "incorrect":
        return not is_correct
    if args.filter_mode == "missing_answer_or_incorrect":
        return (not has_answer) or (not is_correct)
    return False


def load_dataset_split(args: argparse.Namespace):
    print(f"Loading dataset: {args.dataset} from {args.dataset_path}")
    if args.dataset_path and os.path.isdir(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        if args.dataset_config:
            dataset = load_dataset(args.dataset_path, args.dataset_config)
        else:
            dataset = load_dataset(args.dataset_path)

    if args.dataset_config_dict.get("answer_type") == "mmlu-multiple-choice":
        return dataset
    return dataset["train"] if "train" in dataset else dataset["test"]


def init_generator(args: argparse.Namespace) -> Tuple[AutoTokenizer, SLDisaggregationSystem]:
    tokenizer_path = resolve_local_model_path(args.model_path)
    tokenizer_kwargs = {"trust_remote_code": True}
    if tokenizer_path != args.model_path:
        tokenizer_kwargs["local_files_only"] = True
        print(f"Using local tokenizer path: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    router_config = args.model_config.get("router", {})
    switching_strategy = router_config.get("switching_strategy", "neural")
    strategy_kwargs: Dict[str, Any] = {}
    if "js_threshold" in router_config:
        strategy_kwargs["js_threshold"] = router_config["js_threshold"]
    if "js_topk" in router_config:
        strategy_kwargs["js_topk"] = router_config["js_topk"]
    if "entropy_threshold" in router_config:
        strategy_kwargs["entropy_threshold"] = router_config["entropy_threshold"]
    if "router_path" in router_config:
        strategy_kwargs["model_path"] = router_config["router_path"]

    generator = SLDisaggregationSystem(
        model_config=args.model_config,
        device="cuda",
        dtype=torch.bfloat16,
        switching_strategy=switching_strategy,
        strategy_kwargs=strategy_kwargs,
        is_record=args.is_record,
        trace_reference_topk_k=args.trace_reference_topk_k,
        trace_reference_for_all_positions=args.trace_reference_for_all_positions,
        trace_logits_topk_k=args.trace_logits_topk_k,
        quick_sglang_kwargs={
            "dtype": "bfloat16",
            "tp_size": args.slm_tp_size,
            "enable_return_hidden_states": True,
        },
        reference_sglang_kwargs={
            "dtype": "bfloat16",
            "tp_size": args.llm_tp_size,
        },
        overlap_tp_schedule=args.overlap_tp_schedule,
    )
    return tokenizer, generator


def build_continued_prompt(tokenizer: AutoTokenizer, formatted_problem: str, previous_output: str) -> Tuple[str, str]:
    messages = [{"role": "user", "content": formatted_problem}]
    base_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return base_prompt, base_prompt + previous_output


def merge_hybrid_metrics(
    old_row: Dict[str, Any],
    previous_output_tokens: int,
    continuation_output_tokens: int,
    continuation_slm_tokens: int,
    continuation_llm_tokens: int,
    continuation_reference_eval_count: int,
    quick_param: float,
    ref_param: float,
) -> Dict[str, float]:
    old_ref_pct = parse_float(old_row.get("reference_model_percentage"))
    old_ref_eval_count = parse_float(old_row.get("reference_eval_count")) or 0.0

    if old_ref_pct is None:
        total_model_tokens = continuation_slm_tokens + continuation_llm_tokens
        quick_pct = (continuation_slm_tokens / total_model_tokens * 100) if total_model_tokens > 0 else 0.0
        ref_pct = (continuation_llm_tokens / total_model_tokens * 100) if total_model_tokens > 0 else 0.0
        total_params_billions = continuation_slm_tokens * quick_param + continuation_llm_tokens * ref_param
        avg_params_billions = total_params_billions / total_model_tokens if total_model_tokens > 0 else 0.0
        return {
            "quick_model_percentage": quick_pct,
            "reference_model_percentage": ref_pct,
            "reference_eval_count": float(continuation_reference_eval_count),
            "reference_eval_ratio": (
                continuation_reference_eval_count / continuation_output_tokens * 100
                if continuation_output_tokens > 0
                else 0.0
            ),
            "total_params_billions": total_params_billions,
            "avg_params_billions": avg_params_billions,
        }

    old_reference_tokens = old_ref_pct / 100.0 * previous_output_tokens
    old_quick_tokens = max(0.0, previous_output_tokens - old_reference_tokens)

    total_quick_tokens = old_quick_tokens + continuation_slm_tokens
    total_reference_tokens = old_reference_tokens + continuation_llm_tokens
    total_model_tokens = total_quick_tokens + total_reference_tokens
    total_reference_eval_count = old_ref_eval_count + continuation_reference_eval_count
    total_output_tokens = previous_output_tokens + continuation_output_tokens
    total_params_billions = total_quick_tokens * quick_param + total_reference_tokens * ref_param

    return {
        "quick_model_percentage": (
            total_quick_tokens / total_model_tokens * 100 if total_model_tokens > 0 else 0.0
        ),
        "reference_model_percentage": (
            total_reference_tokens / total_model_tokens * 100 if total_model_tokens > 0 else 0.0
        ),
        "reference_eval_count": float(total_reference_eval_count),
        "reference_eval_ratio": (
            total_reference_eval_count / total_output_tokens * 100 if total_output_tokens > 0 else 0.0
        ),
        "total_params_billions": total_params_billions,
        "avg_params_billions": total_params_billions / total_model_tokens if total_model_tokens > 0 else 0.0,
    }


def continue_problems(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    generator: SLDisaggregationSystem,
    problems: List[Dict[str, Any]],
    source_rows: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    temp_dir = os.path.join(args.output_dir, "temp")
    temp_csv_dir = os.path.join(args.output_dir, "temp_csv")
    token_trace_dir = os.path.join(args.output_dir, "token_traces")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_csv_dir, exist_ok=True)
    os.makedirs(token_trace_dir, exist_ok=True)

    dataset_config = args.dataset_config_dict
    answer_type = dataset_config.get("answer_type", "boxed")
    answer_extractor = get_answer_extractor(answer_type)
    quick_param = float(args.model_config.get("quick", {}).get("param", 0))
    ref_param = float(args.model_config.get("reference", {}).get("param", 0))

    results: List[Dict[str, Any]] = []

    for i in range(0, len(problems), args.batch_size):
        batch = problems[i : i + args.batch_size]
        batch_meta = []
        continued_prompts = []
        base_prompts = []

        for item in batch:
            old_row = source_rows[str(item["ID"])]
            previous_output = str(old_row.get("full_output", "") or "")
            base_prompt, continued_prompt = build_continued_prompt(
                tokenizer=tokenizer,
                formatted_problem=item["FormattedProblem"],
                previous_output=previous_output,
            )
            base_prompts.append(base_prompt)
            continued_prompts.append(continued_prompt)
            batch_meta.append((item, old_row, previous_output))

        inputs = [generator.tokenizer.encode(prompt) for prompt in continued_prompts]

        start_time = time.time()
        gen_results = generator.generate(
            inputs,
            max_new_tokens=args.continuation_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        run_time = time.time() - start_time

        continuation_texts = []
        for obj in gen_results:
            output_ids = obj.get("output_ids", []) if isinstance(obj, dict) else obj.output_ids
            continuation_texts.append(generator.tokenizer.decode(output_ids, skip_special_tokens=True))

        batch_output_tokens = [len(tokenizer.encode(text)) for text in continuation_texts]
        total_batch_output_tokens = sum(batch_output_tokens)
        batch_speed = total_batch_output_tokens / run_time if run_time > 0 else None

        for j, ((item, old_row, previous_output), continuation_text, obj) in enumerate(
            zip(batch_meta, continuation_texts, gen_results)
        ):
            combined_output = previous_output + continuation_text
            final_answer = combined_output.split("</think>")[1] if "</think>" in combined_output else combined_output
            predicted_answer, has_answer = answer_extractor(final_answer)
            is_correct = has_answer and check_answer_correctness(predicted_answer, item["Answer"], answer_type)

            previous_output_tokens = parse_int(old_row.get("output_tokens"))
            if previous_output_tokens is None:
                previous_output_tokens = len(tokenizer.encode(previous_output))
            continuation_output_tokens = batch_output_tokens[j]
            total_output_tokens = previous_output_tokens + continuation_output_tokens
            original_prompt_tokens = len(tokenizer.encode(base_prompts[j]))
            continuation_input_tokens = len(tokenizer.encode(continued_prompts[j]))

            slm_token_count = obj.get("slm_token_count", 0) if isinstance(obj, dict) else getattr(obj, "slm_token_count", 0)
            llm_token_count = obj.get("llm_token_count", 0) if isinstance(obj, dict) else getattr(obj, "llm_token_count", 0)
            reference_eval_count = (
                obj.get("reference_eval_count", llm_token_count)
                if isinstance(obj, dict)
                else getattr(obj, "reference_eval_count", llm_token_count)
            )

            metric_bundle = merge_hybrid_metrics(
                old_row=old_row if args.reuse_reference_metrics else {},
                previous_output_tokens=previous_output_tokens,
                continuation_output_tokens=continuation_output_tokens,
                continuation_slm_tokens=slm_token_count,
                continuation_llm_tokens=llm_token_count,
                continuation_reference_eval_count=reference_eval_count,
                quick_param=quick_param,
                ref_param=ref_param,
            )

            result = {
                "problem_id": item["ID"],
                "correct_answer": item["Answer"],
                "has_extracted_answer": has_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "input_tokens": original_prompt_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": original_prompt_tokens + total_output_tokens,
                "full_output": combined_output,
                "run_time": run_time,
                "speed_tokens_per_second": batch_speed,
                "previous_output_tokens": previous_output_tokens,
                "continuation_input_tokens": continuation_input_tokens,
                "continuation_output_tokens": continuation_output_tokens,
                "continuation_max_new_tokens": args.continuation_max_new_tokens,
                "previous_max_new_tokens": args.previous_max_new_tokens,
                "target_max_new_tokens": args.target_max_new_tokens,
                "source_output_dir": args.source_output_dir,
                "source_temp_csv_path": old_row.get("_source_temp_csv_path"),
                "source_has_extracted_answer": old_row.get("has_extracted_answer"),
                "source_is_correct": old_row.get("is_correct"),
                "source_predicted_answer": old_row.get("predicted_answer"),
                "continued_from_existing_output": True,
                "continuation_only_quick_model_percentage": (
                    slm_token_count / (slm_token_count + llm_token_count) * 100
                    if (slm_token_count + llm_token_count) > 0
                    else 0.0
                ),
                "continuation_only_reference_model_percentage": (
                    llm_token_count / (slm_token_count + llm_token_count) * 100
                    if (slm_token_count + llm_token_count) > 0
                    else 0.0
                ),
                "continuation_reference_eval_count": reference_eval_count,
                "model_agreement_percentage": 0,
                "quick_source_agreement_percentage": 0,
            }
            result.update(metric_bundle)

            if dataset_config.get("answer_type") == "multiple_choice" and "Options" in item:
                result["options"] = item["Options"]
            if dataset_config.get("answer_type") == "mmlu-multiple-choice" and "Category" in item:
                result["category"] = item["Category"]

            token_trace = obj.get("token_trace") if isinstance(obj, dict) else None
            if token_trace:
                token_trace_path = os.path.join(token_trace_dir, f"{item['ID']}_run_1.csv")
                save_token_trace_csv(
                    token_trace=token_trace,
                    tokenizer=tokenizer,
                    problem_id=item["ID"],
                    output_path=token_trace_path,
                )
                result["token_trace_path"] = token_trace_path

            temp_output_path = os.path.join(temp_dir, f"{item['ID']}_run_1.txt")
            temp_output_csv_path = os.path.join(temp_csv_dir, f"{item['ID']}_run_1.csv")
            write_to_file(temp_output_path, result)
            write_to_csv(temp_output_csv_path, result)
            results.append(result)

    return results


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(args.model_config, f, indent=2)

    source_rows = load_latest_source_rows(args.source_output_dir)
    selected_ids = {
        str(problem_id)
        for problem_id, row in source_rows.items()
        if should_continue_row(row, args)
    }

    include_ids = parse_problem_id_set(args.problem_ids)
    if include_ids is not None and args.filter_mode != "problem_ids":
        selected_ids &= include_ids

    skip_ids = parse_problem_id_set(args.skip_problem_ids)
    if skip_ids:
        selected_ids -= skip_ids

    dataset_split = load_dataset_split(args)
    all_problems = preprocess_dataset(dataset_split, args.dataset_config_dict, args.output_dir)
    problems = [problem for problem in all_problems if str(problem["ID"]) in selected_ids]

    if args.debug:
        problems = problems[:1]
    elif args.num_problems is not None:
        problems = problems[: args.num_problems]

    print(f"Found {len(source_rows)} source problems with temp_csv rows")
    print(f"Selected {len(selected_ids)} problems after filter: {args.filter_mode}")
    print(f"Running continuation for {len(problems)} problems")

    if not problems:
        print("No problems selected for continuation.")
        return

    tokenizer, generator = init_generator(args)
    try:
        results = continue_problems(args, tokenizer, generator, problems, source_rows)
    finally:
        generator.shutdown()

    model_name = args.model_path.split("/")[-1]
    for result in results:
        result["model_name"] = model_name
        result["model_params"] = args.model_param

    save_results(results, model_name, -1, -1, args.output_dir)
    stats = combine_results(args.output_dir)
    if stats:
        stats_df = pd.DataFrame(list(stats.items()), columns=["metric_name", "value"])
        stats_df.to_csv(os.path.join(args.output_dir, "stats.csv"), index=False)
        print(stats)


if __name__ == "__main__":
    main()

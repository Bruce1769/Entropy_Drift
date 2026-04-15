#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
import time
import urllib.request
from collections import defaultdict


LABELS = [
    "near-equivalent",
    "formatting-only",
    "special-token drift",
    "lexical substitution",
    "math/logic change",
    "clearly divergent",
    "unclear",
]


SYSTEM_PROMPT = (
    "You are a careful judge. Classify the semantic relationship between two candidate next tokens "
    "from different models. You are only given the two tokens, the actual output token, and scalars. "
    "There is no full context, so prefer 'unclear' when unsure. "
    "Return STRICT JSON with keys: label, rationale. Do not use markdown. "
    "Use ONLY these labels: near-equivalent, formatting-only, special-token drift, "
    "lexical substitution, math/logic change, clearly divergent, unclear."
)


def build_user_prompt(row):
    return (
        "Classify the semantic relationship between the two candidate tokens.\n"
        f"SLM top1 token: {row['slm_top1_str']!r}\n"
        f"LLM top1 token: {row['llm_top1_str']!r}\n"
        f"Actual output token: {row['output_token_str']!r}\n"
        f"JS: {row['js']}\n"
        f"SLM entropy: {row['slm_entropy']}\n"
        "If both are mostly the same meaning (case/punct/spacing), use near-equivalent or formatting-only. "
        "If one is a special/template/control token, use special-token drift. "
        "If they are different words but similar meaning, use lexical substitution. "
        "If they imply different math/logic/relations, use math/logic change. "
        "If they are clearly different with no clear relation, use clearly divergent. "
        "If not enough context, use unclear."
    )


def build_batch_prompt(items):
    lines = [
        "Classify each item and return ONLY a JSON array of objects in the SAME ORDER:",
        '[{"label": <label>, "rationale": <short rationale>}].',
        "No markdown, no extra text.",
        "Labels (use exactly):",
        "- near-equivalent: same meaning or trivial case/punct difference",
        "- formatting-only: whitespace/punctuation only",
        "- special-token drift: one is special/control token",
        "- lexical substitution: different words but similar meaning",
        "- math/logic change: implies different math/logic/relations",
        "- clearly divergent: clearly different meanings",
        "- unclear: not enough context",
        "Items:",
    ]
    for idx, item in enumerate(items, 1):
        lines.append(
            f"{idx}. slm_top1={item['slm_top1_str']!r}; "
            f"llm_top1={item['llm_top1_str']!r}; output_token={item['output_token_str']!r}; "
            f"js={item['js']}; slm_entropy={item['slm_entropy']}"
        )
    return "\n".join(lines)


def parse_bool(value):
    return str(value).strip().lower() in ("1", "true", "t", "yes", "y")


def safe_json_from_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def call_deepseek(api_base, api_key, model, messages, timeout=60):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 200,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def safe_text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.encode("utf-8", "replace").decode("utf-8")
    value = value.replace("\x00", "")
    value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", value)
    return value


def stratified_sample(rows, total_target, per_sample_limit, seed):
    rng = random.Random(seed)
    bins = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
    bin_rows = [[] for _ in range(len(bins) - 1)]
    for row in rows:
        js = row["js"]
        for i in range(len(bins) - 1):
            if bins[i] <= js < bins[i + 1]:
                bin_rows[i].append(row)
                break
        else:
            bin_rows[-1].append(row)

    per_bin_target = max(1, total_target // len(bin_rows))
    selected = []
    per_sample_counts = defaultdict(int)
    remaining = []

    for bucket in bin_rows:
        rng.shuffle(bucket)
        count = 0
        for row in bucket:
            if per_sample_counts[row["sample_id"]] >= per_sample_limit:
                remaining.append(row)
                continue
            selected.append(row)
            per_sample_counts[row["sample_id"]] += 1
            count += 1
            if count >= per_bin_target:
                break
        if count < per_bin_target:
            remaining.extend(bucket[count:])

    rng.shuffle(remaining)
    for row in remaining:
        if len(selected) >= total_target:
            break
        if per_sample_counts[row["sample_id"]] >= per_sample_limit:
            continue
        selected.append(row)
        per_sample_counts[row["sample_id"]] += 1

    return selected, bins


def strip_nul_bytes(path):
    with open(path, "rb") as f:
        data = f.read()
    if b"\x00" not in data:
        return
    data = data.replace(b"\x00", b"")
    with open(path, "wb") as f:
        f.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-base", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--per-sample-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api-key or DEEPSEEK_API_KEY.")

    print("[start] loading rows", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "router0_top1diff_semantic_cases.csv")
    summary_json_path = os.path.join(args.output_dir, "router0_top1diff_semantic_summary.json")
    summary_csv_path = os.path.join(args.output_dir, "router0_top1diff_semantic_summary.csv")

    rows = []
    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if parse_bool(row.get("router_decision_replayed", "0")):
                continue
            if parse_bool(row.get("top1_same", "1")):
                continue
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "position": row["position"],
                    "output_token_str": row["output_token_str"],
                    "slm_top1_str": row["slm_top1_str"],
                    "llm_top1_str": row["llm_top1_str"],
                    "slm_entropy": float(row["slm_entropy"]),
                    "js": float(row["js"]),
                }
            )

    print(f"[start] filtered rows: {len(rows)}", flush=True)
    sampled_rows, bins = stratified_sample(
        rows, args.sample_size, args.per_sample_limit, args.seed
    )
    print(f"[start] sampled rows: {len(sampled_rows)}", flush=True)

    counts = defaultdict(int)
    with open(detail_path, "w", newline="") as f:
        fieldnames = [
            "sample_id",
            "position",
            "js",
            "slm_entropy",
            "output_token_str",
            "slm_top1_str",
            "llm_top1_str",
            "label",
            "rationale",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        print("[start] wrote header", flush=True)

        total = len(sampled_rows)
        batch_size = max(1, args.batch_size)
        for start in range(0, total, batch_size):
            batch = sampled_rows[start : start + batch_size]
            items = []
            for row in batch:
                items.append(
                    {
                        "sample_id": row["sample_id"],
                        "position": row["position"],
                        "output_token_str": row["output_token_str"],
                        "slm_top1_str": row["slm_top1_str"],
                        "llm_top1_str": row["llm_top1_str"],
                        "js": row["js"],
                        "slm_entropy": row["slm_entropy"],
                    }
                )
            if start == 0:
                print(f"[call] batch 1 size={len(items)}", flush=True)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_batch_prompt(items)},
            ]

            try:
                response = call_deepseek(
                    args.api_base, args.api_key, args.model, messages
                )
                content = response["choices"][0]["message"]["content"]
                parsed = safe_json_from_response(content)
            except Exception as exc:
                content = None
                parsed = None
                error_note = f"batch_error: {exc}"

            parsed_list = parsed if isinstance(parsed, list) else None
            for idx, item in enumerate(items):
                entry = None
                if parsed_list and idx < len(parsed_list):
                    entry = parsed_list[idx]

                if isinstance(entry, dict):
                    label = entry.get("label", "unclear")
                    rationale = entry.get("rationale", "")
                elif isinstance(entry, str):
                    label = entry
                    rationale = ""
                else:
                    label = "unclear"
                    if content is None:
                        rationale = error_note
                    else:
                        rationale = "missing_label"

                if label not in LABELS:
                    label = "unclear"

                counts[label] += 1
                writer.writerow(
                    {
                        "sample_id": item["sample_id"],
                        "position": item["position"],
                        "js": item["js"],
                        "slm_entropy": item["slm_entropy"],
                        "output_token_str": safe_text(item["output_token_str"]),
                        "slm_top1_str": safe_text(item["slm_top1_str"]),
                        "llm_top1_str": safe_text(item["llm_top1_str"]),
                        "label": label,
                        "rationale": safe_text(rationale),
                    }
                )
                f.flush()

            if args.sleep:
                time.sleep(args.sleep)

            done = min(start + batch_size, total)
            if done % 25 == 0 or done == total:
                print(f"[progress] {done}/{total}", flush=True)

    total = sum(counts.values())
    summary = {
        "total_samples": total,
        "sample_size_requested": args.sample_size,
        "per_sample_limit": args.per_sample_limit,
        "seed": args.seed,
        "js_bins": bins,
        "label_counts": dict(counts),
        "label_ratios": {k: (v / total if total else 0.0) for k, v in counts.items()},
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count", "ratio"])
        for label in LABELS:
            writer.writerow(
                [label, counts.get(label, 0), summary["label_ratios"].get(label, 0.0)]
            )

    strip_nul_bytes(detail_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run fixed long-context quality checks for Experiment 5 KV-cache arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from evaluate_experiment4_quality import check_output
from run_baseline import DEFAULT_MODEL_ID, DEFAULT_MODEL_PATH, ROOT, render_prompt
from run_experiment5_kv_quantization import ARM_SETTINGS, run_once

DEFAULT_FIXTURES = ROOT / "benchmarks" / "fixtures" / "experiment5-quality.json"
DEFAULT_TARGETS = (4096, 8192, 16384)
DISTRACTORS = (
    "The north desk holds pencils and blank paper.",
    "A green folder contains routine maintenance notes.",
    "The afternoon meeting begins after the lunch break.",
    "Window plants are watered every Wednesday.",
    "The storage room has four labeled shelves.",
    "Visitors sign the paper log beside the entrance.",
    "A wall clock hangs above the reception counter.",
    "The south hallway leads to two quiet rooms.",
    "Staff replace notice-board items each month.",
    "A blue binder lists ordinary supply requests.",
)


def allocate_intervals(total: int, positions: list[float]) -> list[int]:
    boundaries = [0.0, *positions, 1.0]
    weights = [boundaries[index + 1] - boundaries[index] for index in range(len(boundaries) - 1)]
    counts = [int(total * weight) for weight in weights]
    for index in range(total - sum(counts)):
        counts[index % len(counts)] += 1
    return counts


def make_quality_fixture(tokenizer, target: int, case: dict) -> dict:
    facts = sorted(case["facts"], key=lambda item: item["position"])
    positions = [item["position"] for item in facts]

    def render_count(filler_count: int):
        intervals = allocate_intervals(filler_count, positions)
        parts: list[str] = []
        distractor_index = 0
        for fact_index, fact in enumerate(facts):
            for _ in range(intervals[fact_index]):
                parts.append(DISTRACTORS[distractor_index % len(DISTRACTORS)])
                distractor_index += 1
            parts.append(fact["text"])
        for _ in range(intervals[-1]):
            parts.append(DISTRACTORS[distractor_index % len(DISTRACTORS)])
            distractor_index += 1
        user = f"Notes:\n{' '.join(parts)}\n\n{case['question']}"
        messages = [
            {"role": "system", "content": case["system"]},
            {"role": "user", "content": user},
        ]
        tokens = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        return tokens, messages

    empty_tokens, _ = render_count(0)
    probe_tokens, _ = render_count(64)
    tokens_per_distractor = max(1.0, (len(probe_tokens) - len(empty_tokens)) / 64)
    estimated = max(1, int((target - len(empty_tokens)) / tokens_per_distractor))
    lower = 0
    upper = estimated + 64
    while len(render_count(upper)[0]) <= target:
        lower = upper
        upper *= 2
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        tokens, _ = render_count(midpoint)
        if len(tokens) <= target:
            lower = midpoint
        else:
            upper = midpoint - 1
    tokens, messages = render_count(lower)
    return {
        "id": case["id"],
        "target_prompt_tokens": target,
        "actual_prompt_tokens": len(tokens),
        "messages": messages,
        "token_ids": tokens,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARM_SETTINGS, required=True)
    parser.add_argument("--quantization-timing", choices=("immediate", "post-prefill"), default="immediate")
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import QuantizedKVCache, make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx.utils import tree_flatten
    import mlx.core as mx

    fixture_path = args.fixtures if args.fixtures.is_absolute() else ROOT / args.fixtures
    fixture_payload = json.loads(fixture_path.read_text())
    arm = ARM_SETTINGS[args.arm]
    model, tokenizer = load(str(DEFAULT_MODEL_PATH))
    sampler = make_sampler(temp=0.0)
    result = {
        "schema_version": 1,
        "experiment": "experiment-5-kv-cache-quantization-quality",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "model": {"id": DEFAULT_MODEL_ID, "local_path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix()},
        "fixture": {
            "path": fixture_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        },
        "settings": {
            "kv_bits": arm["kv_bits"],
            "kv_group_size": arm["kv_group_size"],
            "quantization_timing": args.quantization_timing if arm["kv_bits"] is not None else None,
            "prefill_step_size": 2048,
            "sampling": "greedy",
        },
        "lengths": [],
    }

    total_checks = 0
    passed_checks = 0
    for target in args.targets:
        length_result = {"target_prompt_tokens": target, "cases": []}
        result["lengths"].append(length_result)
        for case in fixture_payload["cases"]:
            fixture = make_quality_fixture(tokenizer, target, case)
            effective_start = (
                fixture["actual_prompt_tokens"]
                if args.quantization_timing == "post-prefill"
                else 0
            )
            run = run_once(
                model,
                tokenizer,
                mx,
                sampler,
                stream_generate,
                make_prompt_cache,
                QuantizedKVCache,
                tree_flatten,
                fixture,
                case["max_tokens"],
                arm["kv_bits"],
                arm["kv_group_size"],
                effective_start,
            )
            evaluation = check_output(run["generated_text"], case["checks"])
            case_checks = len(evaluation["checks"])
            case_passes = sum(check["passed"] for check in evaluation["checks"])
            total_checks += case_checks
            passed_checks += case_passes
            length_result["cases"].append({
                "id": case["id"],
                "actual_prompt_tokens": fixture["actual_prompt_tokens"],
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(fixture["token_ids"], separators=(",", ":")).encode()
                ).hexdigest(),
                "evaluation": evaluation,
                "run": run,
            })
            print(
                f"{args.arm}/{args.quantization_timing} {target} {case['id']}: "
                f"{'PASS' if evaluation['passed'] else 'FAIL'}",
                flush=True,
            )
            write_json(args.output, result)

    result["summary"] = {
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "required_fact_score": passed_checks / total_checks,
        "passed_cases": sum(
            case["evaluation"]["passed"]
            for length in result["lengths"]
            for case in length["cases"]
        ),
        "total_cases": sum(len(length["cases"]) for length in result["lengths"]),
    }
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, result)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()

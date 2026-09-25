#!/usr/bin/env python3
"""Capture repeatable single-user MLX LM inference baselines."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Llama-3.2-1B-Instruct-4bit"
DEFAULT_TARGETS = (64, 512, 2048, 4096)
SYSTEM_PROMPT = "Be concise."
CONTEXT_SENTENCES = (
    "A public library opens at nine each weekday and closes at six.",
    "Visitors can borrow books, use study tables, and attend reading groups.",
    "A small staff maintains the local history collection and answers questions.",
    "The library posts schedule changes at the front desk and on its website.",
    "Community members can suggest new titles through a paper request form.",
    "The meeting room is available for quiet study when no event is scheduled.",
    "The monthly newsletter lists upcoming programs and recent service updates.",
    "The library asks visitors to return borrowed items by the date on each receipt.",
    "Staff help visitors.",
    "Hours change sometimes.",
    "Books are available.",
    "Groups meet monthly.",
    "Updates appear online.",
)

PROMPT_TYPES = {
    "qa": "Answer this question using the notes: What time does the library close on weekdays?",
    "rewrite": "Make this notice friendlier: Return items by the receipt due date.",
    "context_question": "Using the notes, when can visitors use the meeting room for quiet study?",
    "extraction": "From the notes, extract the library's weekday opening and closing times and its available services as compact JSON.",
    "summarization": "Summarize the notes in three concise bullets.",
}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def render_prompt(tokenizer, body: str, request: str) -> tuple[list[int], list[dict[str, str]]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{body}\n\n{request}" if body else request},
    ]
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    return list(tokens), messages


def make_fixture(tokenizer, target: int, prompt_type: str) -> dict:
    """Build a deterministic task-specific prompt close to the requested size."""
    request = PROMPT_TYPES[prompt_type]
    empty_tokens, _ = render_prompt(tokenizer, "", request)
    if len(empty_tokens) >= target:
        raise ValueError(f"Target {target} is shorter than the chat-template overhead")

    probe = " ".join(CONTEXT_SENTENCES * 12)
    probe_tokens, _ = render_prompt(tokenizer, probe, request)
    per_sentence = max(1.0, (len(probe_tokens) - len(empty_tokens)) / 96)
    count = max(0, int((target - len(empty_tokens)) / per_sentence) - 2)
    body_parts = [CONTEXT_SENTENCES[i % len(CONTEXT_SENTENCES)] for i in range(count)]
    tokens, messages = render_prompt(tokenizer, "Notes: " + " ".join(body_parts), request)

    while len(tokens) < target:
        current_body = " ".join(body_parts)
        candidates = []
        for sentence in CONTEXT_SENTENCES:
            candidate_body = f"{current_body} {sentence}" if current_body else f"Notes: {sentence}"
            candidate_tokens, candidate_messages = render_prompt(tokenizer, candidate_body, request)
            candidates.append((len(candidate_tokens), candidate_tokens, candidate_messages, sentence))
        under_target = [candidate for candidate in candidates if candidate[0] <= target]
        if not under_target:
            break
        _, tokens, messages, sentence = max(under_target, key=lambda item: item[0])
        body_parts.append(sentence)

    return {
        "prompt_type": prompt_type,
        "target_prompt_tokens": target,
        "actual_prompt_tokens": len(tokens),
        "messages": messages,
        "token_ids": tokens,
    }


def process_peak_rss_gb() -> float:
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_000_000_000


def run_once(
    model, tokenizer, mx, sampler, stream_generate, fixture, max_new_tokens: int,
    prefill_step_size: int, measure_process_start: bool = False,
) -> dict:
    mx.reset_peak_memory()
    started = time.perf_counter()
    first_token_at = None
    last_response = None
    output_parts: list[str] = []

    for response in stream_generate(
        model,
        tokenizer,
        fixture["token_ids"],
        max_tokens=max_new_tokens,
        prefill_step_size=prefill_step_size,
        sampler=sampler,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        last_response = response
        output_parts.append(response.text)

    finished = time.perf_counter()
    output = "".join(output_parts)
    if last_response is None:
        raise RuntimeError("MLX LM returned no generation response")

    return {
        "prompt_tokens": last_response.prompt_tokens,
        "generated_tokens": last_response.generation_tokens,
        "finish_reason": last_response.finish_reason,
        "time_to_first_token_seconds": first_token_at - started if first_token_at else None,
        "decode_seconds": finished - first_token_at if first_token_at else None,
        "request_seconds": finished - started,
        "process_to_first_token_seconds": (
            first_token_at - SCRIPT_START
            if measure_process_start and first_token_at
            else None
        ),
        "prefill_tokens_per_second": last_response.prompt_tps,
        "decode_tokens_per_second": last_response.generation_tps,
        "mlx_active_memory_gb": mx.get_active_memory() / 1_000_000_000,
        "mlx_peak_memory_gb": mx.get_peak_memory() / 1_000_000_000,
        "process_peak_rss_gb": process_peak_rss_gb(),
        "generated_text": output,
        "generated_text_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def summarize_runs(runs: list[dict]) -> dict:
    """Summarize repeated warm runs with median and nearest-rank p95."""
    metrics = (
        "time_to_first_token_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "request_seconds",
        "mlx_peak_memory_gb",
        "process_peak_rss_gb",
    )
    summary = {"run_count": len(runs)}
    for metric in metrics:
        values = sorted(run[metric] for run in runs if run.get(metric) is not None)
        if values:
            summary[metric] = {
                "median": statistics.median(values),
                "p95_nearest_rank": values[max(0, math.ceil(0.95 * len(values)) - 1)],
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warm-runs", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    args = parser.parse_args()
    if args.warm_runs < 1 or args.max_new_tokens < 1 or args.prefill_step_size < 1:
        parser.error("--warm-runs, --max-new-tokens, and --prefill-step-size must be positive")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {MODEL_PATH}")

    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    started_at = datetime.now(timezone.utc)
    run_stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or ROOT / "benchmarks" / "results" / f"baseline-{run_stamp}.json"
    result = {
        "schema_version": 1,
        "title": "Apple Silicon Inference Engineering: Benchmarking and Optimizing Llama 3.2 1B with MLX LM",
        "started_at_utc": started_at.isoformat(),
        "model": {
            "id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "local_path": MODEL_PATH.relative_to(ROOT).as_posix(),
            "weights_bytes": (MODEL_PATH / "model.safetensors").stat().st_size,
        },
        "environment": {
            "python": sys.version.split()[0],
            "mlx_lm": package_version("mlx-lm"),
            "mlx": package_version("mlx"),
            "transformers": package_version("transformers"),
            "macos": platform.mac_ver()[0],
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "total_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "gpu_utilization_percent": None,
            "gpu_utilization_note": "Not exposed by MLX LM; use Activity Monitor for a visual check.",
        },
        "settings": {
            "chat_template": "tokenizer.apply_chat_template(add_generation_prompt=True)",
            "sampling": "greedy (temperature 0)",
            "max_new_tokens": args.max_new_tokens,
            "prefill_step_size": args.prefill_step_size,
            "concurrency": 1,
            "prompt_cache": "disabled; fresh cache for each request",
            "warm_runs_per_prompt": args.warm_runs,
            "warmup_runs_per_prompt": 1,
            "input_token_targets": args.targets,
            "prompt_types": PROMPT_TYPES,
        },
        "model_load_seconds": None,
        "cold_process_to_first_token_seconds": None,
        "cases": [],
    }

    print(f"Loading {MODEL_PATH}", flush=True)
    load_started = time.perf_counter()
    model, tokenizer = load(str(MODEL_PATH))
    result["model_load_seconds"] = time.perf_counter() - load_started
    sampler = make_sampler(temp=0.0)
    cases_by_length = {target: {"target_prompt_tokens": target, "prompts": []} for target in args.targets}
    result["cases"] = list(cases_by_length.values())
    first_case = True
    for target in args.targets:
        case = cases_by_length[target]
        for prompt_type in PROMPT_TYPES:
            tokenization_started = time.perf_counter()
            fixture = make_fixture(tokenizer, target, prompt_type)
            fixture["fixture_tokenization_seconds"] = time.perf_counter() - tokenization_started
            prompt_result = {
                "prompt_type": prompt_type,
                "target_prompt_tokens": target,
                "actual_prompt_tokens": fixture["actual_prompt_tokens"],
                "fixture_tokenization_seconds": fixture["fixture_tokenization_seconds"],
                "messages": fixture["messages"],
                "cold_first_inference": None,
                "warmup": None,
                "runs": [],
            }
            case["prompts"].append(prompt_result)
            if first_case:
                print(f"Cold first inference: {prompt_type}, {fixture['actual_prompt_tokens']} prompt tokens", flush=True)
                cold = run_once(
                    model, tokenizer, mx, sampler, stream_generate, fixture,
                    args.max_new_tokens, args.prefill_step_size,
                    measure_process_start=True,
                )
                result["cold_process_to_first_token_seconds"] = cold["process_to_first_token_seconds"]
                prompt_result["cold_first_inference"] = cold
                first_case = False
                write_json(output_path, result)

            print(f"Warmup: {prompt_type}, {fixture['actual_prompt_tokens']} prompt tokens", flush=True)
            prompt_result["warmup"] = run_once(
                model, tokenizer, mx, sampler, stream_generate, fixture,
                args.max_new_tokens, args.prefill_step_size,
            )
            write_json(output_path, result)

            for run_number in range(1, args.warm_runs + 1):
                print(f"Run {run_number}/{args.warm_runs}: {prompt_type}, {fixture['actual_prompt_tokens']} prompt tokens", flush=True)
                prompt_result["runs"].append(
                    run_once(
                        model, tokenizer, mx, sampler, stream_generate, fixture,
                        args.max_new_tokens, args.prefill_step_size,
                    )
                )
                write_json(output_path, result)

    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["process_elapsed_seconds"] = time.perf_counter() - SCRIPT_START
    for case in result["cases"]:
        pooled_runs = []
        for prompt_result in case["prompts"]:
            prompt_result["summary"] = summarize_runs(prompt_result["runs"])
            pooled_runs.extend(prompt_result["runs"])
        case["overall_summary_across_prompt_types"] = summarize_runs(pooled_runs)
    write_json(output_path, result)
    print(f"Saved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()

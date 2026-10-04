#!/usr/bin/env python3
"""Benchmark MLX KV-cache quantization for long-context Llama inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_baseline import (
    CONTEXT_SENTENCES,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_PATH,
    PROMPT_TYPES,
    ROOT,
    file_sha256,
    package_version,
    process_peak_rss_gb,
    render_prompt,
)

DEFAULT_TARGETS = (4096, 8192, 16384)
ARM_SETTINGS = {
    "control": {"kv_bits": None, "kv_group_size": None},
    "8bit": {"kv_bits": 8, "kv_group_size": 64},
    "4bit": {"kv_bits": 4, "kv_group_size": 64},
}


def make_long_fixture(tokenizer, target: int, prompt_type: str) -> dict:
    """Build a deterministic long prompt in O(log target) tokenizations."""
    request = PROMPT_TYPES[prompt_type]

    def render_count(sentence_count: int):
        body = " ".join(
            CONTEXT_SENTENCES[index % len(CONTEXT_SENTENCES)]
            for index in range(sentence_count)
        )
        return render_prompt(tokenizer, f"Notes: {body}" if body else "", request)

    empty_tokens, _ = render_count(0)
    probe_tokens, _ = render_count(64)
    tokens_per_sentence = max(1.0, (len(probe_tokens) - len(empty_tokens)) / 64)
    estimated_count = max(1, int((target - len(empty_tokens)) / tokens_per_sentence))
    lower = 0
    upper = estimated_count + 64
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

    sentence_count = lower
    tokens, messages = render_count(sentence_count)
    body_parts = [
        CONTEXT_SENTENCES[index % len(CONTEXT_SENTENCES)]
        for index in range(sentence_count)
    ]

    # Use any remaining token budget without repeatedly rebuilding from scratch.
    while True:
        candidates = []
        for sentence in CONTEXT_SENTENCES:
            body = " ".join([*body_parts, sentence])
            candidate_tokens, candidate_messages = render_prompt(
                tokenizer, f"Notes: {body}", request
            )
            if len(candidate_tokens) <= target:
                candidates.append((len(candidate_tokens), sentence, candidate_tokens, candidate_messages))
        if not candidates:
            break
        _, sentence, tokens, messages = max(candidates, key=lambda item: item[0])
        body_parts.append(sentence)

    return {
        "prompt_type": prompt_type,
        "target_prompt_tokens": target,
        "actual_prompt_tokens": len(tokens),
        "messages": messages,
        "token_ids": tokens,
    }


def nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize_runs(runs: list[dict]) -> dict:
    metrics = (
        "time_to_first_token_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "request_seconds",
        "logical_kv_cache_bytes",
        "allocated_kv_cache_bytes",
        "kv_bytes_per_cached_token",
        "mlx_active_memory_gb",
        "mlx_peak_memory_gb",
        "process_peak_rss_gb",
    )
    summary = {"run_count": len(runs)}
    for metric in metrics:
        values = [run[metric] for run in runs if run.get(metric) is not None]
        if values:
            summary[metric] = {
                "median": statistics.median(values),
                "p95_nearest_rank": nearest_rank_p95(values),
            }
    return summary


def state_nbytes(cache_entry, tree_flatten) -> int:
    return sum(value.nbytes for _, value in tree_flatten(cache_entry.state))


def run_once(
    model,
    tokenizer,
    mx,
    sampler,
    stream_generate,
    make_prompt_cache,
    QuantizedKVCache,
    tree_flatten,
    fixture: dict,
    max_new_tokens: int,
    kv_bits: int | None,
    kv_group_size: int | None,
    quantized_kv_start: int,
) -> dict:
    mx.clear_cache()
    mx.reset_peak_memory()
    prompt_cache = make_prompt_cache(model)
    started = time.perf_counter()
    first_token_at = None
    last_response = None
    output_parts: list[str] = []

    generation_kwargs = {
        "max_tokens": max_new_tokens,
        "prefill_step_size": 2048,
        "sampler": sampler,
        "prompt_cache": prompt_cache,
    }
    if kv_bits is not None:
        generation_kwargs.update(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )

    for response in stream_generate(
        model,
        tokenizer,
        fixture["token_ids"],
        **generation_kwargs,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        last_response = response
        output_parts.append(response.text)

    finished = time.perf_counter()
    if last_response is None:
        raise RuntimeError("MLX LM returned no generation response")

    mx.eval([entry.state for entry in prompt_cache])
    offsets = [entry.offset for entry in prompt_cache]
    if len(set(offsets)) != 1:
        raise RuntimeError(f"KV-cache layer offsets diverged: {offsets}")
    cached_tokens = offsets[0]
    logical_bytes = sum(state_nbytes(entry, tree_flatten) for entry in prompt_cache)
    allocated_bytes = sum(entry.nbytes for entry in prompt_cache)
    quantized_layers = sum(isinstance(entry, QuantizedKVCache) for entry in prompt_cache)
    output = "".join(output_parts)

    return {
        "prompt_tokens": last_response.prompt_tokens,
        "generated_tokens": last_response.generation_tokens,
        "finish_reason": last_response.finish_reason,
        "time_to_first_token_seconds": first_token_at - started,
        "decode_seconds": finished - first_token_at,
        "request_seconds": finished - started,
        "prefill_tokens_per_second": last_response.prompt_tps,
        "decode_tokens_per_second": last_response.generation_tps,
        "cached_tokens": cached_tokens,
        "cache_layer_count": len(prompt_cache),
        "quantized_cache_layers": quantized_layers,
        "quantized_cache_layer_fraction": quantized_layers / len(prompt_cache),
        "quantized_kv_start_effective": quantized_kv_start if kv_bits is not None else None,
        "cache_types": sorted({type(entry).__name__ for entry in prompt_cache}),
        "logical_kv_cache_bytes": logical_bytes,
        "allocated_kv_cache_bytes": allocated_bytes,
        "kv_bytes_per_cached_token": logical_bytes / cached_tokens,
        "mlx_active_memory_gb": mx.get_active_memory() / 1_000_000_000,
        "mlx_peak_memory_gb": mx.get_peak_memory() / 1_000_000_000,
        "process_peak_rss_gb": process_peak_rss_gb(),
        "generated_text": output,
        "generated_text_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARM_SETTINGS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-runs", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=tuple(PROMPT_TYPES),
        default=tuple(PROMPT_TYPES),
    )
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument(
        "--quantization-timing",
        choices=("immediate", "post-prefill"),
        default="immediate",
        help="Quantize immediately or after the prompt has been prefetched",
    )
    args = parser.parse_args()

    if args.warm_runs < 1 or args.max_new_tokens < 1:
        parser.error("--warm-runs and --max-new-tokens must be positive")
    if any(target < 1 for target in args.targets):
        parser.error("all --targets must be positive")
    if args.quantized_kv_start < 0:
        parser.error("--quantized-kv-start must be non-negative")
    if not DEFAULT_MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {DEFAULT_MODEL_PATH}")

    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import QuantizedKVCache, make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx.utils import tree_flatten
    import mlx.core as mx

    arm = ARM_SETTINGS[args.arm]
    started_at = datetime.now(timezone.utc)
    weight_paths = sorted(DEFAULT_MODEL_PATH.glob("*.safetensors"))
    result = {
        "schema_version": 1,
        "experiment": "experiment-5-kv-cache-quantization",
        "started_at_utc": started_at.isoformat(),
        "arm": args.arm,
        "model": {
            "id": DEFAULT_MODEL_ID,
            "local_path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(),
            "weights_bytes": sum(path.stat().st_size for path in weight_paths),
            "weight_files": [
                {"name": path.name, "sha256": file_sha256(path)}
                for path in weight_paths
            ],
        },
        "environment": {
            "python": sys.version.split()[0],
            "mlx_lm": package_version("mlx-lm"),
            "mlx": package_version("mlx"),
        },
        "settings": {
            "kv_bits": arm["kv_bits"],
            "kv_group_size": arm["kv_group_size"],
            "quantization_timing": (
                args.quantization_timing if arm["kv_bits"] is not None else None
            ),
            "quantized_kv_start": (
                "prompt_length"
                if arm["kv_bits"] is not None and args.quantization_timing == "post-prefill"
                else args.quantized_kv_start if arm["kv_bits"] is not None else None
            ),
            "prefill_step_size": 2048,
            "sampling": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "concurrency": 1,
            "fresh_cache_per_request": True,
            "warmup_runs_per_prompt": 1,
            "measured_runs_per_prompt": args.warm_runs,
            "targets": args.targets,
            "prompt_types": args.prompt_types,
        },
        "model_load_seconds": None,
        "cases": [],
    }

    load_started = time.perf_counter()
    model, tokenizer = load(str(DEFAULT_MODEL_PATH))
    result["model_load_seconds"] = time.perf_counter() - load_started
    sampler = make_sampler(temp=0.0)

    for target in args.targets:
        length_case = {"target_prompt_tokens": target, "prompts": []}
        result["cases"].append(length_case)
        for prompt_type in args.prompt_types:
            fixture = make_long_fixture(tokenizer, target, prompt_type)
            effective_quantized_kv_start = (
                fixture["actual_prompt_tokens"]
                if args.quantization_timing == "post-prefill"
                else args.quantized_kv_start
            )
            token_bytes = json.dumps(fixture["token_ids"], separators=(",", ":")).encode()
            prompt_result = {
                "prompt_type": prompt_type,
                "target_prompt_tokens": target,
                "actual_prompt_tokens": fixture["actual_prompt_tokens"],
                "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
                "messages": fixture["messages"],
                "warmup": None,
                "runs": [],
            }
            length_case["prompts"].append(prompt_result)
            print(
                f"{args.arm} warmup: {prompt_type}, {fixture['actual_prompt_tokens']} prompt tokens",
                flush=True,
            )
            prompt_result["warmup"] = run_once(
                model,
                tokenizer,
                mx,
                sampler,
                stream_generate,
                make_prompt_cache,
                QuantizedKVCache,
                tree_flatten,
                fixture,
                args.max_new_tokens,
                arm["kv_bits"],
                arm["kv_group_size"],
                effective_quantized_kv_start,
            )
            write_json(args.output, result)

            for run_number in range(1, args.warm_runs + 1):
                print(
                    f"{args.arm} run {run_number}/{args.warm_runs}: "
                    f"{prompt_type}, {fixture['actual_prompt_tokens']} prompt tokens",
                    flush=True,
                )
                prompt_result["runs"].append(
                    run_once(
                        model,
                        tokenizer,
                        mx,
                        sampler,
                        stream_generate,
                        make_prompt_cache,
                        QuantizedKVCache,
                        tree_flatten,
                        fixture,
                        args.max_new_tokens,
                        arm["kv_bits"],
                        arm["kv_group_size"],
                        effective_quantized_kv_start,
                    )
                )
                write_json(args.output, result)

    for length_case in result["cases"]:
        pooled_runs = []
        for prompt_result in length_case["prompts"]:
            prompt_result["summary"] = summarize_runs(prompt_result["runs"])
            pooled_runs.extend(prompt_result["runs"])
        length_case["overall_summary_across_prompt_types"] = summarize_runs(pooled_runs)

    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, result)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()

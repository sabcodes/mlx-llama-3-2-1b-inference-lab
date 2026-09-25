#!/usr/bin/env python3
"""Capture repeatable single-user MLX LM inference baselines."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Llama-3.2-1B-Instruct-4bit"
DEFAULT_TARGETS = (64, 512, 2048, 4096)
SYSTEM_PROMPT = "Be concise."
REQUEST = "Summarize the context in three bullets."
CONTEXT_SENTENCES = (
    "A public library opens at nine each weekday and closes at six.",
    "Visitors can borrow books, use study tables, and attend reading groups.",
    "A small staff maintains the local history collection and answers questions.",
    "The library posts schedule changes at the front desk and on its website.",
    "Community members can suggest new titles through a paper request form.",
    "The meeting room is available for quiet study when no event is scheduled.",
    "The monthly newsletter lists upcoming programs and recent service updates.",
    "The library asks visitors to return borrowed items by the date on each receipt.",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def render_prompt(tokenizer, body: str) -> tuple[list[int], list[dict[str, str]]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{body}\n\n{REQUEST}" if body else REQUEST},
    ]
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    return list(tokens), messages


def make_fixture(tokenizer, target: int) -> dict:
    """Build a deterministic chat prompt close to the requested token count."""
    empty_tokens, _ = render_prompt(tokenizer, "")
    if len(empty_tokens) >= target:
        raise ValueError(f"Target {target} is shorter than the chat-template overhead")

    probe = " ".join(CONTEXT_SENTENCES * 12)
    probe_tokens, _ = render_prompt(tokenizer, probe)
    per_sentence = max(1.0, (len(probe_tokens) - len(empty_tokens)) / 96)
    count = max(0, int((target - len(empty_tokens)) / per_sentence) - 2)
    body_parts = [CONTEXT_SENTENCES[i % len(CONTEXT_SENTENCES)] for i in range(count)]
    tokens, messages = render_prompt(tokenizer, " ".join(body_parts))

    while len(tokens) < target:
        current_body = " ".join(body_parts)
        candidates = []
        for sentence in CONTEXT_SENTENCES:
            candidate_body = f"{current_body} {sentence}" if current_body else sentence
            candidate_tokens, candidate_messages = render_prompt(tokenizer, candidate_body)
            candidates.append((len(candidate_tokens), candidate_tokens, candidate_messages, sentence))
        under_target = [candidate for candidate in candidates if candidate[0] <= target]
        if not under_target:
            break
        _, tokens, messages, sentence = max(under_target, key=lambda item: item[0])
        body_parts.append(sentence)

    return {
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
    measure_process_start: bool = False,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warm-runs", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    args = parser.parse_args()
    if args.warm_runs < 1 or args.max_new_tokens < 1:
        parser.error("--warm-runs and --max-new-tokens must be positive")
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
            "local_path": str(MODEL_PATH),
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
            "prefill_step_size": 2048,
            "concurrency": 1,
            "prompt_cache": "disabled; fresh cache for each request",
            "warm_runs_per_case": args.warm_runs,
            "warmup_runs_per_case": 1,
            "input_token_targets": args.targets,
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
    fixtures = []
    for target in args.targets:
        tokenization_started = time.perf_counter()
        fixture = make_fixture(tokenizer, target)
        fixture["fixture_tokenization_seconds"] = time.perf_counter() - tokenization_started
        fixtures.append(fixture)

    for case_index, fixture in enumerate(fixtures):
        case = {
            "target_prompt_tokens": fixture["target_prompt_tokens"],
            "actual_prompt_tokens": fixture["actual_prompt_tokens"],
            "fixture_tokenization_seconds": fixture["fixture_tokenization_seconds"],
            "messages": fixture["messages"],
            "cold_first_inference": None,
            "warmup": None,
            "runs": [],
        }
        result["cases"].append(case)

        if case_index == 0:
            print(f"Cold first inference: {fixture['actual_prompt_tokens']} prompt tokens", flush=True)
            cold = run_once(
                model, tokenizer, mx, sampler, stream_generate, fixture,
                args.max_new_tokens, measure_process_start=True,
            )
            result["cold_process_to_first_token_seconds"] = cold["process_to_first_token_seconds"]
            case["cold_first_inference"] = cold
            write_json(output_path, result)

        print(f"Warmup: {fixture['actual_prompt_tokens']} prompt tokens", flush=True)
        case["warmup"] = run_once(model, tokenizer, mx, sampler, stream_generate, fixture, args.max_new_tokens)
        write_json(output_path, result)

        for run_number in range(1, args.warm_runs + 1):
            print(f"Run {run_number}/{args.warm_runs}: {fixture['actual_prompt_tokens']} prompt tokens", flush=True)
            case["runs"].append(
                run_once(model, tokenizer, mx, sampler, stream_generate, fixture, args.max_new_tokens)
            )
            write_json(output_path, result)

    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["process_elapsed_seconds"] = time.perf_counter() - SCRIPT_START
    write_json(output_path, result)
    print(f"Saved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()

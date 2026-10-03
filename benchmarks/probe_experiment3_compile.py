#!/usr/bin/env python3
"""Bounded feasibility probe for explicit MLX compilation of single-token decode."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Llama-3.2-1B-Instruct-4bit"
FIXTURE_PATH = ROOT / "benchmarks" / "fixtures" / "experiment2-sessions.json"
MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def process_peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def render_prompt(tokenizer) -> list[int]:
    source = json.loads(FIXTURE_PATH.read_text())
    fixture = next(item for item in source["scenarios"] if item["id"] == "library")
    notes = list(fixture["notes"])
    counter = 0
    while True:
        system_prompt = (
            f"Reference file: {fixture['title']}. Answer from these notes.\n"
            + "\n".join(f"- {line}" for line in notes)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": fixture["turns"][0]["user"]},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if isinstance(rendered, dict):
            rendered = rendered["input_ids"]
        tokens = [int(token) for token in rendered]
        if len(tokens) >= 512:
            return tokens
        notes.append(
            f"Supplemental record {counter}: {fixture['notes'][counter % len(fixture['notes'])]}"
        )
        counter += 1


def cache_bytes(states) -> int:
    total = 0
    for state in states:
        for value in state:
            total += int(value.nbytes)
    return total


def prefill(model, prompt_tokens, make_prompt_cache, mx):
    cache = make_prompt_cache(model)
    prompt = mx.array(prompt_tokens, dtype=mx.uint32)
    processed = 0
    while len(prompt) - processed > 1:
        count = min(2048, (len(prompt) - processed) - 1)
        model(prompt[processed : processed + count][None], cache=cache)
        mx.eval([item.state for item in cache])
        processed += count
    return cache, prompt[-1:]


def eager_decode(model, cache, current_token, count, mx):
    token_ids = []
    latencies = []
    mx.reset_peak_memory()
    for _ in range(count):
        started = time.perf_counter()
        logits = model(current_token[None], cache=cache)[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        next_token = mx.argmax(logprobs, axis=-1)
        mx.eval(next_token, [item.state for item in cache])
        latencies.append(time.perf_counter() - started)
        token_ids.append(int(next_token.item()))
        current_token = next_token
    states = [item.state for item in cache]
    return {
        "token_ids": token_ids,
        "latencies_seconds": latencies,
        "first_token_seconds": latencies[0],
        "median_subsequent_token_seconds": statistics.median(latencies[1:]),
        "subsequent_tokens_per_second": 1.0 / statistics.median(latencies[1:]),
        "cache_bytes": cache_bytes(states),
        "cache_tokens": int(cache[0].size()),
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }


def compiled_decode(model, cache, current_token, count, mx, *, shapeless: bool):
    cache_types = [type(item) for item in cache]
    meta_states = [item.meta_state for item in cache]
    states = [item.state for item in cache]
    trace_calls = 0

    def decode_step(current_token, current_states):
        nonlocal trace_calls
        trace_calls += 1
        local_cache = [
            cache_type.from_state(state, meta_state)
            for cache_type, state, meta_state in zip(
                cache_types, current_states, meta_states
            )
        ]
        logits = model(current_token[None], cache=local_cache)[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        next_token = mx.argmax(logprobs, axis=-1)
        return next_token, [item.state for item in local_cache]

    compiled_step = mx.compile(
        decode_step,
        inputs=model.state,
        shapeless=shapeless,
    )

    token_ids = []
    latencies = []
    mx.reset_peak_memory()
    for _ in range(count):
        started = time.perf_counter()
        current_token, states = compiled_step(current_token, states)
        mx.eval(current_token, states)
        latencies.append(time.perf_counter() - started)
        token_ids.append(int(current_token.item()))

    return {
        "token_ids": token_ids,
        "latencies_seconds": latencies,
        "first_token_seconds_including_compile": latencies[0],
        "median_subsequent_token_seconds": statistics.median(latencies[1:]),
        "subsequent_tokens_per_second": 1.0 / statistics.median(latencies[1:]),
        "python_trace_calls": trace_calls,
        "cache_bytes": cache_bytes(states),
        "cache_tokens": int(states[0][0].shape[-2]),
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compile-mode", choices=("shapeless", "shaped"), default="shapeless"
    )
    args = parser.parse_args()
    if args.tokens < 2 or args.tokens > 64:
        parser.error("--tokens must be between 2 and 64 for this bounded probe")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {MODEL_PATH}")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    started = datetime.now(timezone.utc)
    output = args.output or (
        ROOT
        / "benchmarks"
        / "results"
        / f"experiment3-compile-probe-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    if not output.is_absolute():
        output = ROOT / output
    result = {
        "schema_version": 1,
        "experiment": "experiment-3-compiled-decode",
        "stage": "bounded-feasibility-probe",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": None,
        "model": {"id": MODEL_ID, "local_path": MODEL_PATH.relative_to(ROOT).as_posix()},
        "environment": {
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            "architecture": platform.machine(),
            "mlx": package_version("mlx"),
            "mlx_lm": package_version("mlx-lm"),
        },
        "settings": {
            "prompt_profile_tokens": 512,
            "decode_tokens": args.tokens,
            "prefill_step_size": 2048,
            "sampling": "greedy",
            "compiled_scope": "single-token model, log-softmax, and argmax",
            "compile_mode": args.compile_mode,
            "compile_shapeless": args.compile_mode == "shapeless",
            "bounded_probe_only": True,
        },
        "eager": None,
        "compiled": None,
        "comparison": None,
        "error": None,
    }
    atomic_write_json(output, result)

    model, tokenizer = load(str(MODEL_PATH))
    prompt_tokens = render_prompt(tokenizer)
    result["actual_prompt_tokens"] = len(prompt_tokens)

    eager_cache, eager_token = prefill(model, prompt_tokens, make_prompt_cache, mx)
    result["eager"] = eager_decode(
        model, eager_cache, eager_token, args.tokens, mx
    )
    atomic_write_json(output, result)

    try:
        compiled_cache, compiled_token = prefill(
            model, prompt_tokens, make_prompt_cache, mx
        )
        result["compiled"] = compiled_decode(
            model,
            compiled_cache,
            compiled_token,
            args.tokens,
            mx,
            shapeless=args.compile_mode == "shapeless",
        )
        eager_ids = result["eager"]["token_ids"]
        compiled_ids = result["compiled"]["token_ids"]
        first_difference = next(
            (
                index
                for index, pair in enumerate(zip(eager_ids, compiled_ids))
                if pair[0] != pair[1]
            ),
            None,
        )
        exact_match = eager_ids == compiled_ids
        result["comparison"] = {
            "exact_token_match": exact_match,
            "first_different_token_index": first_difference,
            "steady_decode_speedup_fraction": (
                result["compiled"]["subsequent_tokens_per_second"]
                / result["eager"]["subsequent_tokens_per_second"]
                - 1.0
            ),
            "peak_mlx_memory_increase_fraction": (
                result["compiled"]["mlx_peak_memory_bytes"]
                / result["eager"]["mlx_peak_memory_bytes"]
                - 1.0
            ),
            "cache_token_count_match": (
                result["compiled"]["cache_tokens"]
                == result["eager"]["cache_tokens"]
            ),
            "bounded_feasibility_pass": (
                exact_match
                and result["compiled"]["python_trace_calls"] <= 2
                and result["compiled"]["cache_tokens"]
                == result["eager"]["cache_tokens"]
            ),
        }
    except Exception as error:  # Preserve evidence from an unsupported compile path.
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error).replace(chr(58) * 2, ":"),
        }

    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(output, result)
    print(output.relative_to(ROOT))
    print(json.dumps({"comparison": result["comparison"], "error": result["error"]}, indent=2))


if __name__ == "__main__":
    main()

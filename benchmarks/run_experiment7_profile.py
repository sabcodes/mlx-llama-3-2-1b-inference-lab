#!/usr/bin/env python3
"""Profile MLX-LM request phases and optionally record a focused Metal trace."""

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
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_PATH,
    ROOT,
    make_fixture,
    package_version,
    process_peak_rss_gb,
    project_relative_path,
    write_json,
)

DEFAULT_TARGETS = (64, 512, 2048, 4096)
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "experiment7-phase-profile-20261004.json"


def nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def artifact_size_bytes(path: Path) -> int:
    """Return logical bytes without double-counting trace symlinks."""
    if path.is_file():
        return path.stat().st_size
    files = [
        item
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    ]
    return sum(item.stat().st_size for item in files)


def summarize_metric(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "p95_nearest_rank": nearest_rank_p95(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_runs(runs: list[dict]) -> dict:
    metrics = (
        "time_to_first_token_seconds",
        "decode_seconds",
        "request_seconds",
        "measured_decode_tokens_per_second",
        "inter_token_latency_median_seconds",
        "inter_token_latency_p95_seconds",
        "ttft_share_of_request",
        "decode_share_of_request",
        "mlx_peak_memory_gb",
        "process_peak_rss_gb",
    )
    summary = {"run_count": len(runs)}
    for metric in metrics:
        values = [float(run[metric]) for run in runs if run.get(metric) is not None]
        if values:
            summary[metric] = summarize_metric(values)
    return summary


def collect_timing(
    *,
    model,
    tokenizer,
    mx,
    stream_generate,
    make_prompt_cache,
    sampler,
    logits_processors,
    token_ids: list[int],
    max_tokens: int,
    prefill_step_size: int,
) -> dict:
    prompt_cache = make_prompt_cache(model)
    mx.reset_peak_memory()
    started = time.perf_counter()
    response_times: list[float] = []
    output_token_ids: list[int] = []
    last_response = None

    for response in stream_generate(
        model,
        tokenizer,
        token_ids,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
        sampler=sampler,
        logits_processors=logits_processors,
    ):
        response_times.append(time.perf_counter())
        output_token_ids.append(int(response.token))
        last_response = response

    finished = time.perf_counter()
    if last_response is None or not response_times:
        raise RuntimeError("MLX-LM returned no generation response")

    ttft = response_times[0] - started
    request_seconds = finished - started
    decode_seconds = finished - response_times[0]
    decode_transitions = max(0, len(response_times) - 1)
    inter_token = [
        current - previous
        for previous, current in zip(response_times, response_times[1:])
    ]
    token_bytes = json.dumps(output_token_ids, separators=(",", ":")).encode()

    return {
        "prompt_tokens": int(last_response.prompt_tokens),
        "generated_tokens": int(last_response.generation_tokens),
        "finish_reason": last_response.finish_reason,
        "time_to_first_token_seconds": ttft,
        "decode_seconds": decode_seconds,
        "request_seconds": request_seconds,
        "mlx_reported_prompt_tokens_per_second": float(last_response.prompt_tps),
        "mlx_reported_generation_tokens_per_second": float(last_response.generation_tps),
        "measured_decode_tokens_per_second": (
            decode_transitions / decode_seconds
            if decode_transitions and decode_seconds > 0
            else None
        ),
        "inter_token_latency_median_seconds": (
            statistics.median(inter_token) if inter_token else None
        ),
        "inter_token_latency_p95_seconds": (
            nearest_rank_p95(inter_token) if inter_token else None
        ),
        "ttft_share_of_request": ttft / request_seconds,
        "decode_share_of_request": decode_seconds / request_seconds,
        "mlx_active_memory_gb": mx.get_active_memory() / 1_000_000_000,
        "mlx_peak_memory_gb": mx.get_peak_memory() / 1_000_000_000,
        "process_peak_rss_gb": process_peak_rss_gb(),
        "output_token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
    }


def run_timing(args) -> None:
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    import mlx.core as mx

    model_path = args.model_path.resolve()
    model_relative_path = project_relative_path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")

    started_at = datetime.now(timezone.utc)
    load_started = time.perf_counter()
    model, tokenizer = load(str(model_path))
    load_seconds = time.perf_counter() - load_started
    sampler = make_sampler(temp=0.0)
    logits_processors = make_logits_processors(
        logit_bias={token: -1e9 for token in tokenizer.eos_token_ids}
    )

    result = {
        "schema_version": 1,
        "experiment": "experiment-7-runtime-phase-profile",
        "started_at_utc": started_at.isoformat(),
        "model": {
            "id": DEFAULT_MODEL_ID,
            "local_path": model_relative_path,
            "weights_bytes": sum(
                path.stat().st_size for path in model_path.glob("*.safetensors")
            ),
            "load_seconds": load_seconds,
        },
        "environment": {
            "python": sys.version.split()[0],
            "mlx": package_version("mlx"),
            "mlx_lm": package_version("mlx-lm"),
            "metal_device": mx.device_info(),
        },
        "settings": {
            "sampling": "greedy",
            "eos_suppressed": True,
            "concurrency": 1,
            "fresh_cache_per_request": True,
            "prefill_step_size": args.prefill_step_size,
            "prompt_type": "summarization",
            "target_prompt_tokens": args.targets,
            "max_new_tokens": args.max_tokens,
            "warmups_per_case": args.warmups,
            "measured_runs_per_case": args.repeats,
            "clean_timing_without_powermetrics": True,
        },
        "measurement_notes": {
            "ttft": "Wall time through receipt of the first generated token; includes prompt prefill, first-token model work, and sampling.",
            "decode": "Wall time after the first response through the final response.",
            "measured_decode_tps": "Generated-token transitions after the first token divided by decode wall time.",
            "inter_token_latency": "Wall-clock intervals between successive streamed token responses.",
        },
        "cases": [],
    }

    for target in args.targets:
        fixture_started = time.perf_counter()
        fixture = make_fixture(tokenizer, target, "summarization")
        fixture_seconds = time.perf_counter() - fixture_started
        fixture_bytes = json.dumps(
            fixture["token_ids"], separators=(",", ":")
        ).encode()
        case = {
            "id": f"{target}_summarization_forced_{args.max_tokens}",
            "target_prompt_tokens": target,
            "actual_prompt_tokens": fixture["actual_prompt_tokens"],
            "prompt_token_ids_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
            "fixture_construction_seconds": fixture_seconds,
            "warmups": [],
            "runs": [],
        }
        result["cases"].append(case)
        kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "mx": mx,
            "stream_generate": stream_generate,
            "make_prompt_cache": make_prompt_cache,
            "sampler": sampler,
            "logits_processors": logits_processors,
            "token_ids": fixture["token_ids"],
            "max_tokens": args.max_tokens,
            "prefill_step_size": args.prefill_step_size,
        }
        for warmup in range(1, args.warmups + 1):
            print(f"warmup {warmup}/{args.warmups}: {case['id']}", flush=True)
            case["warmups"].append(collect_timing(**kwargs))
            write_json(args.output, result)
        for run in range(1, args.repeats + 1):
            print(f"run {run}/{args.repeats}: {case['id']}", flush=True)
            case["runs"].append(collect_timing(**kwargs))
            write_json(args.output, result)
        case["summary"] = summarize_runs(case["runs"])
        write_json(args.output, result)

    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["conclusion_inputs"] = {
        "dominant_phase_by_case": {
            case["id"]: (
                "ttft"
                if case["summary"]["ttft_share_of_request"]["median"] >= 0.5
                else "decode"
            )
            for case in result["cases"]
        },
        "profiling_gate": (
            "A phase is dominant when its median share is at least 50%. "
            "A custom-kernel experiment requires a Metal trace to identify an "
            "operation family with enough end-to-end share for a plausible 10% win."
        ),
    }
    write_json(args.output, result)
    print(f"Saved timing profile to {args.output}", flush=True)


def consume_direct_generation(
    *, model, mx, generate_step, make_prompt_cache, sampler, logits_processors,
    token_ids: list[int], max_tokens: int, prefill_step_size: int,
) -> list[int]:
    generator = generate_step(
        mx.array(token_ids),
        model,
        max_tokens=max_tokens,
        prompt_cache=make_prompt_cache(model),
        prefill_step_size=prefill_step_size,
        sampler=sampler,
        logits_processors=logits_processors,
    )
    tokens = [int(token) for token, _ in generator]
    mx.synchronize()
    return tokens


def run_capture(args) -> None:
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("Launch with MTL_CAPTURE_ENABLED=1 to enable Metal capture")
    if args.trace_output is None:
        raise ValueError("--trace-output is required in capture mode")

    trace_path = args.trace_output.resolve()
    try:
        trace_relative = trace_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Trace output must stay inside this project") from error
    if trace_path.suffix != ".gputrace":
        raise ValueError("Trace output must use the .gputrace extension")
    if trace_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing trace: {trace_path}")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    import mlx.core as mx

    model_path = args.model_path.resolve()
    project_relative_path(model_path)
    model, tokenizer = load(str(model_path))
    fixture = make_fixture(tokenizer, args.capture_target, "summarization")
    sampler = make_sampler(temp=0.0)
    logits_processors = make_logits_processors(
        logit_bias={token: -1e9 for token in tokenizer.eos_token_ids}
    )
    common = {
        "model": model,
        "mx": mx,
        "generate_step": generate_step,
        "make_prompt_cache": make_prompt_cache,
        "sampler": sampler,
        "logits_processors": logits_processors,
        "token_ids": fixture["token_ids"],
        "prefill_step_size": args.prefill_step_size,
    }

    # Compile/load all normal kernels before recording so the trace represents
    # steady inference rather than one-time startup work.
    consume_direct_generation(max_tokens=8, **common)

    generator = generate_step(
        mx.array(fixture["token_ids"]),
        model,
        max_tokens=args.capture_decode_tokens + 8,
        prompt_cache=make_prompt_cache(model),
        prefill_step_size=args.prefill_step_size,
        sampler=sampler,
        logits_processors=logits_processors,
    )
    captured_tokens: list[int] = []
    capture_started = False
    capture_start = None
    try:
        if args.capture_phase == "decode":
            # Prime prefill plus several decode steps, then synchronize. The
            # generator keeps one token of look-ahead, noted in the metadata.
            for _ in range(4):
                next(generator)
            mx.synchronize()
        mx.metal.start_capture(str(trace_path))
        capture_started = True
        capture_start = time.perf_counter()
        steps = 1 if args.capture_phase == "prefill" else args.capture_decode_tokens
        for _ in range(steps):
            token, _ = next(generator)
            captured_tokens.append(int(token))
        mx.synchronize()
        capture_seconds = time.perf_counter() - capture_start
        mx.metal.stop_capture()
        capture_started = False
    finally:
        if capture_started:
            mx.metal.stop_capture()

    metadata_path = args.output
    trace_logical_bytes = artifact_size_bytes(trace_path)
    token_bytes = json.dumps(captured_tokens, separators=(",", ":")).encode()
    metadata = {
        "schema_version": 1,
        "experiment": "experiment-7-metal-capture",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_phase": args.capture_phase,
        "trace_path": trace_relative,
        "trace_logical_bytes": trace_logical_bytes,
        "model": {"id": DEFAULT_MODEL_ID, "local_path": project_relative_path(model_path)},
        "environment": {
            "mlx": package_version("mlx"),
            "mlx_lm": package_version("mlx-lm"),
            "metal_device": mx.device_info(),
        },
        "settings": {
            "actual_prompt_tokens": fixture["actual_prompt_tokens"],
            "prefill_step_size": args.prefill_step_size,
            "captured_decode_steps": (
                0 if args.capture_phase == "prefill" else args.capture_decode_tokens
            ),
            "warmup_completed_before_capture": True,
        },
        "capture_seconds": capture_seconds,
        "captured_token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "limitations": [
            "The prefill capture includes first-token work and may include one queued decode look-ahead operation.",
            "The decode generator pipelines one token ahead, so capture boundaries are approximate.",
            "Capture wall time includes heavy Metal recording overhead and is not an inference benchmark.",
            "Kernel-level interpretation requires Xcode Metal Debugger; raw .gputrace files are gitignored.",
        ],
    }
    write_json(metadata_path, metadata)
    print(f"Saved Metal trace to {trace_path}", flush=True)
    print(f"Saved capture metadata to {metadata_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("timing", "capture"), default="timing")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--capture-phase", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--capture-target", type=int, default=2048)
    parser.add_argument("--capture-decode-tokens", type=int, default=16)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()

    if not args.model_path.is_absolute():
        args.model_path = ROOT / args.model_path
    if args.output is None:
        if args.mode == "capture":
            parser.error("--output is required in capture mode")
        args.output = DEFAULT_OUTPUT
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    if args.trace_output is not None and not args.trace_output.is_absolute():
        args.trace_output = ROOT / args.trace_output
    positive = (
        args.warmups,
        args.repeats,
        args.max_tokens,
        args.prefill_step_size,
        args.capture_target,
        args.capture_decode_tokens,
        *args.targets,
    )
    if any(value < 1 for value in positive):
        parser.error("all numeric run settings must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "timing":
        run_timing(args)
    else:
        run_capture(args)


if __name__ == "__main__":
    main()

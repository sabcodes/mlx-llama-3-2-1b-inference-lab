#!/usr/bin/env python3
"""Run one separate-process arm of Experiment 6's bounded performance screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_baseline import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_PATH,
    ROOT,
    file_sha256,
    make_fixture,
    package_version,
    project_relative_path,
)
from run_experiment6_correctness import (
    DEFAULT_DRAFT_PATH,
    collect_generation,
    structural_signature,
    write_json,
)

SCRIPT_START = time.perf_counter()
DEFAULT_TARGETS = (64, 512, 4096)
WORKLOADS = (
    {"id": "short_qa", "prompt_type": "qa", "max_tokens": 32, "suppress_eos": False},
    {
        "id": "long_summarization",
        "prompt_type": "summarization",
        "max_tokens": 128,
        "suppress_eos": True,
    },
)


def nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize_runs(runs: list[dict]) -> dict:
    metrics = (
        "time_to_first_token_seconds",
        "request_seconds",
        "decode_tokens_per_second",
        "mlx_active_memory_gb",
        "mlx_peak_memory_gb",
        "process_peak_rss_gb",
    )
    summary = {"run_count": len(runs)}
    for metric in metrics:
        values = [run[metric] for run in runs]
        summary[metric] = {
            "median": statistics.median(values),
            "p95_nearest_rank": nearest_rank_p95(values),
        }
    if runs[0].get("draft_metrics") is not None:
        for metric in (
            "draft_origin_fraction",
            "reconstructed_acceptance_fraction",
            "accepted_per_verification_round",
            "verification_rounds",
        ):
            values = [run["draft_metrics"][metric] for run in runs]
            summary[metric] = {
                "median": statistics.median(values),
                "p95_nearest_rank": nearest_rank_p95(values),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("target", "zero-draft", "draft"), required=True)
    parser.add_argument("--draft-count", type=int)
    parser.add_argument("--target-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--draft-model-path", type=Path, default=DEFAULT_DRAFT_PATH)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.arm == "draft" and (args.draft_count is None or args.draft_count < 1):
        parser.error("--draft-count must be positive for --arm draft")
    if args.arm != "draft" and args.draft_count is not None:
        parser.error("--draft-count is valid only for --arm draft")
    if args.warm_runs < 1 or args.prefill_step_size < 1:
        parser.error("--warm-runs and --prefill-step-size must be positive")
    if any(target < 1 for target in args.targets):
        parser.error("all --targets must be positive")

    target_path = args.target_model_path if args.target_model_path.is_absolute() else ROOT / args.target_model_path
    draft_path = args.draft_model_path if args.draft_model_path.is_absolute() else ROOT / args.draft_model_path
    target_path = target_path.resolve()
    draft_path = draft_path.resolve()
    project_relative_path(target_path)
    project_relative_path(draft_path)
    if not target_path.is_dir():
        raise FileNotFoundError(f"Target model directory not found: {target_path}")
    if args.arm != "target" and not draft_path.is_dir():
        raise FileNotFoundError(f"Draft model directory not found: {draft_path}")

    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    import mlx.core as mx

    started_at = datetime.now(timezone.utc)
    target_load_started = time.perf_counter()
    model, tokenizer = load(str(target_path))
    target_load_seconds = time.perf_counter() - target_load_started
    active_after_target_load = mx.get_active_memory() / 1_000_000_000
    draft_model = None
    draft_load_seconds = None
    active_after_both_load = None
    if args.arm != "target":
        target_tokenizer_hash = file_sha256(target_path / "tokenizer.json")
        draft_tokenizer_hash = file_sha256(draft_path / "tokenizer.json")
        if target_tokenizer_hash != draft_tokenizer_hash:
            raise ValueError("Target and draft tokenizer.json files are not byte-identical")
        if structural_signature(target_path / "config.json") != structural_signature(
            draft_path / "config.json"
        ):
            raise ValueError("Target and draft structural configurations differ")
        draft_load_started = time.perf_counter()
        draft_model, draft_tokenizer = load(str(draft_path))
        draft_load_seconds = time.perf_counter() - draft_load_started
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError("Target and draft vocabulary sizes differ")
        active_after_both_load = mx.get_active_memory() / 1_000_000_000

    effective_draft_count = (
        None if args.arm == "target" else 0 if args.arm == "zero-draft" else args.draft_count
    )
    result = {
        "schema_version": 1,
        "experiment": "experiment-6-speculative-decoding-stage-1",
        "started_at_utc": started_at.isoformat(),
        "arm": args.arm,
        "draft_count": effective_draft_count,
        "environment": {
            "python": sys.version.split()[0],
            "mlx": package_version("mlx"),
            "mlx_lm": package_version("mlx-lm"),
        },
        "target": {
            "id": DEFAULT_MODEL_ID,
            "local_path": project_relative_path(target_path),
            "weights_bytes": sum(path.stat().st_size for path in target_path.glob("*.safetensors")),
            "load_seconds": target_load_seconds,
        },
        "draft": None,
        "settings": {
            "sampling": "greedy",
            "prefill_step_size": args.prefill_step_size,
            "concurrency": 1,
            "fresh_cache_per_request": True,
            "warmup_runs_per_fixture": 1,
            "measured_runs_per_fixture": args.warm_runs,
            "targets": args.targets,
            "workloads": WORKLOADS,
        },
        "memory_after_load": {
            "target_only_active_gb": active_after_target_load,
            "target_and_draft_active_gb": active_after_both_load,
        },
        "cases": [],
    }
    if draft_model is not None:
        result["draft"] = {
            "id": draft_path.name,
            "local_path": project_relative_path(draft_path),
            "weights_bytes": sum(path.stat().st_size for path in draft_path.glob("*.safetensors")),
            "load_seconds": draft_load_seconds,
        }

    sampler = make_sampler(temp=0.0)
    first_request = True
    for target in args.targets:
        for workload in WORKLOADS:
            fixture = make_fixture(tokenizer, target, workload["prompt_type"])
            logit_bias = (
                {token: -1e9 for token in tokenizer.eos_token_ids}
                if workload["suppress_eos"]
                else None
            )
            logits_processors = make_logits_processors(logit_bias=logit_bias)
            token_bytes = json.dumps(fixture["token_ids"], separators=(",", ":")).encode()
            case = {
                "id": f"{target}_{workload['id']}",
                "target_prompt_tokens": target,
                "actual_prompt_tokens": fixture["actual_prompt_tokens"],
                "prompt_type": workload["prompt_type"],
                "max_tokens": workload["max_tokens"],
                "suppress_eos": workload["suppress_eos"],
                "prompt_token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
                "warmup": None,
                "runs": [],
            }
            result["cases"].append(case)
            kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "mx": mx,
                "sampler": sampler,
                "stream_generate": stream_generate,
                "make_prompt_cache": make_prompt_cache,
                "token_ids": fixture["token_ids"],
                "max_tokens": workload["max_tokens"],
                "prefill_step_size": args.prefill_step_size,
                "logits_processors": logits_processors,
                "draft_model": draft_model,
                "draft_count": effective_draft_count,
            }
            print(f"{args.arm}/{effective_draft_count} warmup: {case['id']}", flush=True)
            warmup_started = time.perf_counter()
            case["warmup"] = collect_generation(**kwargs)
            if first_request:
                result["cold_process_to_first_token_seconds"] = (
                    warmup_started
                    - SCRIPT_START
                    + case["warmup"]["time_to_first_token_seconds"]
                )
                first_request = False
            write_json(args.output, result)
            for run_number in range(1, args.warm_runs + 1):
                print(
                    f"{args.arm}/{effective_draft_count} run {run_number}/{args.warm_runs}: {case['id']}",
                    flush=True,
                )
                case["runs"].append(collect_generation(**kwargs))
                write_json(args.output, result)
            case["summary"] = summarize_runs(case["runs"])

    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["process_elapsed_seconds"] = time.perf_counter() - SCRIPT_START
    write_json(args.output, result)
    print(f"Saved Stage-1 arm to {args.output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure multi-turn exact KV-prefix reuse against full prompt recomputation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Llama-3.2-1B-Instruct-4bit"
FIXTURE_PATH = ROOT / "benchmarks" / "fixtures" / "experiment2-sessions.json"
MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
PREFILL_STEP_SIZE = 2048


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirty": "30",
}


def normalize_fact_text(value: str) -> str:
    """Normalize simple wording differences used by the factual fixture checks."""
    words = re.findall(r"[a-z0-9]+", value.casefold())
    return " ".join(NUMBER_WORDS.get(word, word) for word in words)


def contains_expected_fact(output_text: str, expected_fact: str) -> bool:
    return normalize_fact_text(expected_fact) in normalize_fact_text(output_text)


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def render_conversation(tokenizer, system_prompt: str, messages: list[dict[str, str]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, *messages],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    if rendered and hasattr(rendered[0], "ids"):
        rendered = rendered[0].ids
    return [int(token) for token in rendered]


def build_fixtures(tokenizer) -> tuple[dict, list[dict]]:
    source = json.loads(FIXTURE_PATH.read_text())
    built = []
    for fixture in source["scenarios"]:
        notes = fixture["notes"]
        target = int(fixture["profile_tokens"])
        expanded = list(notes)
        system_prompt = ""
        first_tokens = []
        counter = 0
        while True:
            system_prompt = (
                f"Reference file: {fixture['title']}. Answer from these notes.\n"
                + "\n".join(f"- {line}" for line in expanded)
            )
            first_messages = [{"role": "user", "content": fixture["turns"][0]["user"]}]
            first_tokens = render_conversation(tokenizer, system_prompt, first_messages)
            if len(first_tokens) >= target:
                break
            note = notes[counter % len(notes)]
            counter += 1
            expanded.append(f"Supplemental record {counter}: {note}")
            if counter > 1000:
                raise RuntimeError(f"Could not construct {target}-token context for {fixture['id']}")

        built.append({
            "id": fixture["id"],
            "profile_tokens": target,
            "title": fixture["title"],
            "system_prompt": system_prompt,
            "system_prompt_sha256": digest_text(system_prompt),
            "first_turn_prompt_tokens": len(first_tokens),
            "turns": fixture["turns"],
        })
    return source, built


def process_peak_rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def nearest_rank_p95(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    return values[max(0, math.ceil(0.95 * len(values)) - 1)]


def summarize_rows(rows: list[dict], metric: str) -> dict | None:
    values = [row[metric] for row in rows if row.get(metric) is not None]
    if not values:
        return None
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95_nearest_rank": nearest_rank_p95(values),
    }


def session_key(session: dict) -> tuple[str, str, int]:
    return (session["fixture_id"], "warmup" if session["warmup"] else "measured", session["repetition"])


def load_baseline(path: Path) -> dict[tuple[str, str, int], dict]:
    result = json.loads(path.read_text())
    if result.get("arm") != "control_full_recompute" or result.get("completed_at_utc") is None:
        raise ValueError("Baseline must be a completed Experiment 2 full-recompute control JSON")
    return {session_key(session): session for session in result["sessions"]}


def validate_matching_baseline(baseline: dict, treatment: dict) -> None:
    if baseline.get("fixture_set", {}).get("sha256") != treatment["fixture_set"]["sha256"]:
        raise ValueError("Treatment fixtures do not match the control baseline")
    for field in ("prefill_step_size", "max_new_tokens", "warmups_per_fixture",
                  "measured_repetitions_per_fixture", "measurement_mode",
                  "smoke_validation_only"):
        if baseline.get("settings", {}).get(field) != treatment["settings"].get(field):
            raise ValueError(f"Baseline setting mismatch: {field}")
    if baseline.get("model", {}).get("id") != treatment["model"]["id"]:
        raise ValueError("Treatment model does not match the control baseline")


def create_result(args, started_at: datetime, source: dict, fixtures: list[dict]) -> dict:
    fixture_payload = [
        {key: value for key, value in fixture.items() if key != "system_prompt"}
        | {"system_prompt_sha256": fixture["system_prompt_sha256"]}
        for fixture in fixtures
    ]
    fixture_hash = hashlib.sha256(
        json.dumps(fixture_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "experiment": "experiment-2-kv-prefix-reuse",
        "protocol_version": "1.1" if args.correctness_mode == "practical" else "1.0",
        "git_revision": git_revision(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arm": "control_full_recompute" if args.arm == "control" else "treatment_prefix_reuse",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": None,
        "model": {
            "id": MODEL_ID,
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
        },
        "settings": {
            "prefill_step_size": args.prefill_step_size,
            "sampling": "greedy (temperature 0)",
            "max_new_tokens": args.max_new_tokens,
            "concurrency": 1,
            "warmups_per_fixture": args.warmups,
            "measured_repetitions_per_fixture": args.repeats,
            "measurement_mode": args.measurement_mode,
            "cache_scope": "fresh per turn" if args.arm == "control" else "session-local exact token prefix",
            "cache_reset_per_session": True,
            "telemetry": "off" if args.measurement_mode == "timing_only" else "powermetrics instrumented",
            "smoke_validation_only": args.smoke,
            "correctness_mode": args.correctness_mode,
        },
        "fixture_set": {
            "id": source["fixture_set_id"],
            "sha256": fixture_hash,
            "fixtures": fixture_payload,
        },
        "baseline_json": project_relative(args.baseline_json) if args.baseline_json else None,
        "telemetry_log": project_relative(args.telemetry_log) if args.telemetry_log else None,
        "sessions": [],
        "summary": {},
    }


def run_session(
    *, fixture: dict, repetition: int, warmup: bool, args, model, tokenizer,
    mx, stream_generate, make_sampler, make_prompt_cache, LRUPromptCache,
    baseline_session: dict | None, on_update,
) -> dict:
    messages: list[dict[str, str]] = []
    session = {
        "fixture_id": fixture["id"],
        "profile_tokens": fixture["profile_tokens"],
        "repetition": repetition,
        "warmup": warmup,
        "reset_before_session": True,
        "cache_reset_verified": args.arm == "reuse",
        "valid_for_paired_analysis": True if args.arm == "reuse" else None,
        "turns": [],
    }
    cache_index = LRUPromptCache(max_size=16) if args.arm == "reuse" else None
    if cache_index is not None and len(cache_index) != 0:
        raise AssertionError("Session cache was not empty at its reset boundary")
    model_key = MODEL_ID
    sampler = make_sampler(temp=0.0)
    session_started = time.perf_counter()
    on_update(session)

    for turn_index, turn in enumerate(fixture["turns"], start=1):
        turn_started = time.perf_counter()
        messages.append({"role": "user", "content": turn["user"]})
        full_tokens = render_conversation(tokenizer, fixture["system_prompt"], messages)
        render_finished = time.perf_counter()
        cache_lookup_started = time.perf_counter()
        if args.arm == "control":
            prompt_cache = make_prompt_cache(model)
            suffix = full_tokens
            cached_tokens = 0
            cache_status = "fresh"
        else:
            prompt_cache, suffix = cache_index.fetch_nearest_cache(model_key, full_tokens)
            cached_tokens = len(full_tokens) - len(suffix)
            if prompt_cache is None:
                prompt_cache = make_prompt_cache(model)
                cache_status = "miss"
                if suffix != full_tokens:
                    raise AssertionError("Cache miss returned an unexpected suffix")
            else:
                cache_status = "hit" if cached_tokens else "miss"
            if turn_index == 1 and cache_status != "miss":
                raise AssertionError("New session did not begin with an empty-cache miss")
            if not suffix:
                raise AssertionError("Prefix matcher returned an empty generation suffix")
        cache_lookup_finished = time.perf_counter()
        cache_logical_before = [int(item.size()) for item in prompt_cache]

        mx.reset_peak_memory()
        started = time.perf_counter()
        first_at = None
        output_parts: list[str] = []
        output_token_ids: list[int] = []
        final = None
        for response in stream_generate(
            model,
            tokenizer,
            suffix,
            max_tokens=args.max_new_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
            prefill_step_size=args.prefill_step_size,
        ):
            if first_at is None:
                first_at = time.perf_counter()
            output_parts.append(response.text)
            output_token_ids.append(int(response.token))
            final = response
        finished = time.perf_counter()
        if final is None or first_at is None:
            raise RuntimeError("MLX-LM returned no generation response")
        output_text = "".join(output_parts)
        output_bytes = output_text.encode("utf-8")

        expected_ids = None
        output_equivalent = None
        output_diff = None
        baseline_fact_check_passed = None
        practical_quality_screen_passed = None
        if baseline_session is not None:
            if turn_index > len(baseline_session["turns"]):
                raise ValueError("Baseline session has fewer turns than the fixture")
            baseline_turn = baseline_session["turns"][turn_index - 1]
            expected_ids = baseline_turn["generated_token_ids"]
            output_equivalent = expected_ids == output_token_ids
            if not output_equivalent:
                first_diff = next(
                    (i for i, (control_id, reuse_id) in enumerate(zip(expected_ids, output_token_ids))
                     if control_id != reuse_id),
                    min(len(expected_ids), len(output_token_ids)),
                )
                output_diff = {
                    "first_different_token_index": first_diff,
                    "control_token_count": len(expected_ids),
                    "reuse_token_count": len(output_token_ids),
                    "control_token_at_divergence": expected_ids[first_diff] if first_diff < len(expected_ids) else None,
                    "reuse_token_at_divergence": output_token_ids[first_diff] if first_diff < len(output_token_ids) else None,
                }

            baseline_fact_check_passed = contains_expected_fact(
                baseline_turn["output_text"], turn["expected_fact"]
            )

        fact_check_passed = contains_expected_fact(output_text, turn["expected_fact"])
        if baseline_session is not None:
            practical_quality_screen_passed = not (
                baseline_fact_check_passed and not fact_check_passed
            )
            if args.correctness_mode == "exact" and not output_equivalent:
                session["valid_for_paired_analysis"] = False
            elif args.correctness_mode == "practical" and not practical_quality_screen_passed:
                session["valid_for_paired_analysis"] = False
        cache_bytes = sum(int(item.nbytes) for item in prompt_cache)
        turn_result = {
            "turn_index": turn_index,
            "user_prompt": turn["user"],
            "expected_fact": turn["expected_fact"],
            "fact_check_passed": fact_check_passed,
            "baseline_fact_check_passed": baseline_fact_check_passed,
            "practical_quality_screen_passed": practical_quality_screen_passed,
            "input_tokens": len(full_tokens),
            "cached_prefix_tokens": cached_tokens,
            "suffix_tokens_processed": len(suffix),
            "cache_status": cache_status,
            "cache_bytes": cache_bytes,
            "cache_logical_tokens_before": cache_logical_before,
            "cache_logical_tokens_after": [int(item.size()) for item in prompt_cache],
            "cache_key_tokens_after": len(full_tokens) + len(output_token_ids),
            "prompt_tps_suffix_basis": final.prompt_tps,
            "decode_tokens_per_second": final.generation_tps,
            "render_seconds": render_finished - turn_started,
            "cache_lookup_seconds": cache_lookup_finished - cache_lookup_started,
            "generation_time_to_first_token_seconds": first_at - started,
            "time_to_first_token_seconds": first_at - turn_started,
            "request_seconds": finished - started,
            "turn_end_to_end_seconds": finished - turn_started,
            "generation_tokens": len(output_token_ids),
            "finish_reason": final.finish_reason,
            "mlx_active_memory_bytes": int(mx.get_active_memory()),
            "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "generated_token_ids": output_token_ids,
            "output_text": output_text,
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "output_equivalent_to_control": output_equivalent,
            "output_divergence": output_diff,
        }
        session["turns"].append(turn_result)

        if args.arm == "reuse":
            cache_key = list(full_tokens) + output_token_ids
            cache_index.insert_cache(model_key, cache_key, prompt_cache)
            turn_result["session_cache_entries"] = len(cache_index)
            turn_result["session_cache_bytes"] = int(cache_index.nbytes)

        messages.append({"role": "assistant", "content": output_text})
        on_update(session)
        if (
            baseline_session is not None
            and args.correctness_mode == "exact"
            and not output_equivalent
        ):
            session["stop_reason"] = "generated token IDs diverged from control; downstream turns omitted"
            break

    session["session_seconds"] = time.perf_counter() - session_started
    session["session_max_mlx_peak_memory_bytes"] = max(
        (turn["mlx_peak_memory_bytes"] for turn in session["turns"]), default=0
    )
    session["session_max_cache_bytes"] = max(
        (turn.get("session_cache_bytes", 0) for turn in session["turns"]), default=0
    )
    if warmup and baseline_session is not None and session["valid_for_paired_analysis"] is False:
        reason = (
            "generated token IDs diverged from control"
            if args.correctness_mode == "exact"
            else "automated factual quality regressed from control"
        )
        raise RuntimeError(
            f"Warmup correctness failed for fixture {fixture['id']}: {reason}; stop before timing"
        )
    on_update(session)
    return session


def make_summary(result: dict) -> dict:
    grouped = defaultdict(list)
    for session in result["sessions"]:
        if session["warmup"]:
            continue
        for turn in session["turns"]:
            grouped[(session["fixture_id"], turn["turn_index"])].append(turn)
    per_turn = {}
    for (fixture_id, turn_index), rows in sorted(grouped.items()):
        per_turn[f"{fixture_id}:turn-{turn_index}"] = {
            metric: summarize_rows(rows, metric)
            for metric in (
                "input_tokens", "cached_prefix_tokens", "suffix_tokens_processed",
                "prompt_tps_suffix_basis", "render_seconds", "cache_lookup_seconds",
                "generation_time_to_first_token_seconds", "time_to_first_token_seconds",
                "request_seconds", "turn_end_to_end_seconds",
                "decode_tokens_per_second", "mlx_active_memory_bytes",
                "mlx_peak_memory_bytes", "cache_bytes", "session_cache_bytes",
            )
            if summarize_rows(rows, metric) is not None
        }
        per_turn[f"{fixture_id}:turn-{turn_index}"]["fact_check_pass_rate"] = (
            sum(1 for row in rows if row["fact_check_passed"]) / len(rows)
        )
        equivalents = [row["output_equivalent_to_control"] for row in rows]
        if any(value is not None for value in equivalents):
            per_turn[f"{fixture_id}:turn-{turn_index}"]["output_equivalence_rate"] = (
                sum(1 for value in equivalents if value is True)
                / sum(1 for value in equivalents if value is not None)
            )
    result_summary = {
        "measured_session_count": sum(not session["warmup"] for session in result["sessions"]),
        "measured_turn_count": sum(
            len(session["turns"]) for session in result["sessions"] if not session["warmup"]
        ),
        "valid_paired_sessions": sum(
            1 for session in result["sessions"]
            if not session["warmup"] and session["valid_for_paired_analysis"] is not False
        ),
        "per_fixture_turn": per_turn,
    }
    return result_summary


def add_paired_decision(
    result: dict, baseline_sessions: dict[tuple[str, str, int], dict], correctness_mode: str
) -> None:
    changes = []
    memory = []
    mismatch_count = 0
    practical_screen_failure_count = 0
    for session in result["sessions"]:
        if session["warmup"]:
            continue
        key = session_key(session)
        control = baseline_sessions.get(key)
        if control is None:
            continue
        if session["valid_for_paired_analysis"] is False:
            practical_screen_failure_count += 1
        for treatment_turn, control_turn in zip(session["turns"], control["turns"]):
            if treatment_turn.get("output_equivalent_to_control") is not True:
                mismatch_count += 1
            if correctness_mode == "exact" and treatment_turn.get("output_equivalent_to_control") is not True:
                continue
            if correctness_mode == "practical" and treatment_turn.get("practical_quality_screen_passed") is not True:
                continue
            if treatment_turn["cached_prefix_tokens"] > 0:
                changes.append(
                    (control_turn["time_to_first_token_seconds"] - treatment_turn["time_to_first_token_seconds"])
                    / control_turn["time_to_first_token_seconds"]
                )
        if control["session_max_mlx_peak_memory_bytes"]:
            memory.append(
                session["session_max_mlx_peak_memory_bytes"]
                / control["session_max_mlx_peak_memory_bytes"] - 1
            )
    median_gain = statistics.median(changes) if changes else None
    median_memory_increase = statistics.median(memory) if memory else None
    promising = (
        median_gain is not None
        and median_gain >= 0.10
        and median_memory_increase is not None
        and median_memory_increase <= 0.10
        and (mismatch_count == 0 if correctness_mode == "exact" else practical_screen_failure_count == 0)
    )
    if promising and correctness_mode == "practical":
        outcome = "performance_and_automated_quality_screen_pass;_manual_semantic_review_required"
    elif promising:
        outcome = "promising"
    else:
        outcome = "not_promising_or_insufficient_valid_pairs"
    result["paired_decision"] = {
        "paired_cache_hit_turn_count": len(changes),
        "median_ttft_improvement_fraction": median_gain,
        "p95_ttft_improvement_fraction": nearest_rank_p95(changes),
        "median_session_peak_mlx_memory_increase_fraction": median_memory_increase,
        "output_mismatch_turn_count": mismatch_count,
        "practical_quality_screen_failure_session_count": practical_screen_failure_count,
        "correctness_mode": correctness_mode,
        "criterion": (
            "promising if median hit-turn TTFT gain >= 10%, median session peak MLX memory "
            "increase <= 10%, and no exact output mismatch"
            if correctness_mode == "exact"
            else "candidate if median hit-turn TTFT gain >= 10%, median session peak MLX memory "
            "increase <= 10%, no loss of required facts versus control, and manual review finds "
            "no material correctness, instruction-following, or usefulness regression"
        ),
        "outcome": outcome,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "reuse"), default="control")
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=PREFILL_STEP_SIZE)
    parser.add_argument("--measurement-mode", choices=("timing_only", "telemetry_instrumented"), default="timing_only")
    parser.add_argument("--telemetry-log", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run only one short fixture for correctness validation")
    parser.add_argument("--fixture-id", help="Restrict a diagnostic/smoke run to one fixture ID")
    parser.add_argument(
        "--correctness-mode",
        choices=("exact", "practical"),
        default="exact",
        help="Use exact token equality or a factual screen followed by manual semantic review",
    )
    args = parser.parse_args()
    if min(args.warmups, args.repeats, args.max_new_tokens, args.prefill_step_size) < 1:
        parser.error("run counts, token cap, and prefill step size must be positive")
    if args.arm == "reuse" and args.baseline_json is None:
        parser.error("--baseline-json is required for --arm reuse")
    if args.arm == "control" and args.baseline_json is not None:
        parser.error("--baseline-json is only used for --arm reuse")
    if args.prefill_step_size != PREFILL_STEP_SIZE:
        parser.error("Experiment 2 holds prefill step size fixed at 2048")
    if args.measurement_mode == "telemetry_instrumented" and args.telemetry_log is None:
        parser.error("--telemetry-log is required for instrumented runs")
    if args.measurement_mode == "timing_only" and args.telemetry_log is not None:
        parser.error("--telemetry-log requires --measurement-mode telemetry_instrumented")
    if args.telemetry_log is not None and not args.telemetry_log.exists():
        parser.error("Telemetry log does not exist yet; start powermetrics before this run")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {MODEL_PATH}")

    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    baseline_document = json.loads(args.baseline_json.read_text()) if args.baseline_json else None
    baseline_sessions = load_baseline(args.baseline_json) if args.baseline_json else {}
    load_started = time.perf_counter()
    model, tokenizer = load(str(MODEL_PATH))
    model_load_seconds = time.perf_counter() - load_started
    source, fixtures = build_fixtures(tokenizer)
    if args.fixture_id:
        fixtures = [fixture for fixture in fixtures if fixture["id"] == args.fixture_id]
        if not fixtures:
            parser.error(f"unknown fixture ID: {args.fixture_id}")
    if args.smoke:
        fixtures = fixtures[:1]
        args.warmups = 1
        args.repeats = 1
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    prefix = "experiment2-smoke" if args.smoke else "experiment2"
    output = args.output or ROOT / "benchmarks" / "results" / f"{prefix}-{args.arm}-{stamp}.json"
    result = create_result(args, started_at, source, fixtures)
    result["model_load_seconds"] = model_load_seconds
    if baseline_document is not None:
        validate_matching_baseline(baseline_document, result)
    atomic_write_json(output, result)

    for fixture in fixtures:
        # One session warmup, then the requested measured sessions. Every session gets
        # a fresh treatment cache so no state leaks between repetitions.
        run_specs = [(True, i + 1) for i in range(args.warmups)] + [
            (False, i + 1) for i in range(args.repeats)
        ]
        for warmup, repetition in run_specs:
            baseline_session = None
            if args.arm == "reuse":
                key = (fixture["id"], "warmup" if warmup else "measured", repetition)
                baseline_session = baseline_sessions.get(key)
                if baseline_session is None:
                    raise ValueError(f"No matching control session in baseline for {key}")
            def update_session(current: dict) -> None:
                key = session_key(current)
                for index, existing in enumerate(result["sessions"]):
                    if session_key(existing) == key:
                        result["sessions"][index] = current
                        break
                else:
                    result["sessions"].append(current)
                result["summary"] = make_summary(result)
                atomic_write_json(output, result)

            session = run_session(
                fixture=fixture,
                repetition=repetition,
                warmup=warmup,
                args=args,
                model=model,
                tokenizer=tokenizer,
                mx=mx,
                stream_generate=stream_generate,
                make_sampler=make_sampler,
                make_prompt_cache=make_prompt_cache,
                LRUPromptCache=LRUPromptCache,
                baseline_session=baseline_session,
                on_update=update_session,
            )
            result["summary"] = make_summary(result)
            atomic_write_json(output, result)
            print(
                f"{fixture['id']} {'warmup' if warmup else f'repeat {repetition}'}: "
                f"{len(session['turns'])} turns, paired-valid={session['valid_for_paired_analysis']}"
            )
            if args.arm == "reuse" and session["valid_for_paired_analysis"] is False and warmup:
                raise RuntimeError(f"Warmup output mismatch for {fixture['id']}; timing aborted")

    if args.arm == "reuse":
        add_paired_decision(result, baseline_sessions, args.correctness_mode)
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["summary"] = make_summary(result)
    atomic_write_json(output, result)
    print(f"Completed: {project_relative(output)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Experiment 6's exact-token speculative-decoding correctness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
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

DEFAULT_DRAFT_PATH = ROOT / "models" / "Llama-3.2-1B-Instruct-mixed3_6-g64-exp4"
DEFAULT_FIXTURES = ROOT / "benchmarks" / "fixtures" / "experiment6-stage0.json"
STRUCTURAL_CONFIG_KEYS = (
    "architectures",
    "bos_token_id",
    "eos_token_id",
    "head_dim",
    "hidden_size",
    "intermediate_size",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def process_peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_000_000_000


def structural_signature(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    return {key: config.get(key) for key in STRUCTURAL_CONFIG_KEYS}


def cache_summary(entries: list) -> dict:
    offsets = [int(entry.offset) for entry in entries]
    return {
        "layer_count": len(entries),
        "cache_types": sorted({type(entry).__name__ for entry in entries}),
        "offsets": offsets,
        "unique_offsets": sorted(set(offsets)),
        "offsets_consistent": len(set(offsets)) == 1,
        "allocated_bytes": sum(int(entry.nbytes) for entry in entries),
    }


def draft_cycle_metrics(flags: list[bool], draft_count: int, max_tokens: int) -> dict:
    """Reconstruct proposal opportunities from emitted draft/target flags."""
    accepted = sum(flags)
    target_emitted = len(flags) - accepted
    proposed = 0
    rounds = 0
    emitted = 0
    index = 0
    while index < len(flags):
        rounds += 1
        proposed += min(draft_count, max_tokens - emitted)
        while index < len(flags) and flags[index]:
            emitted += 1
            index += 1
        if index < len(flags):
            emitted += 1
            index += 1
    return {
        "accepted_draft_tokens": accepted,
        "target_emitted_tokens": target_emitted,
        "verification_rounds": rounds,
        "reconstructed_proposal_opportunities": proposed,
        "accepted_per_verification_round": accepted / rounds if rounds else 0.0,
        "draft_origin_fraction": accepted / len(flags) if flags else 0.0,
        "reconstructed_acceptance_fraction": accepted / proposed if proposed else 0.0,
        "proposal_metric_note": (
            "Proposal opportunities are reconstructed from emitted from_draft flags; "
            "an EOS inside a verification block can leave later proposals unobserved."
        ),
    }


def first_difference(left: list[int], right: list[int]) -> dict | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return {
                "position_zero_based": index,
                "control_token": left_token,
                "candidate_token": right_token,
            }
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "position_zero_based": index,
            "control_token": left[index] if index < len(left) else None,
            "candidate_token": right[index] if index < len(right) else None,
        }
    return None


def collect_generation(
    *,
    model,
    tokenizer,
    mx,
    sampler,
    stream_generate,
    make_prompt_cache,
    token_ids: list[int],
    max_tokens: int,
    prefill_step_size: int,
    logits_processors: list,
    draft_model=None,
    draft_count: int | None = None,
) -> dict:
    target_layer_count = len(model.layers)
    prompt_cache = make_prompt_cache(model)
    if draft_model is not None:
        prompt_cache.extend(make_prompt_cache(draft_model))

    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    first_token_at = None
    responses = []
    kwargs = {
        "max_tokens": max_tokens,
        "prefill_step_size": prefill_step_size,
        "sampler": sampler,
        "prompt_cache": prompt_cache,
        "logits_processors": logits_processors,
    }
    if draft_model is not None:
        kwargs.update(draft_model=draft_model, num_draft_tokens=draft_count)

    for response in stream_generate(model, tokenizer, token_ids, **kwargs):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        responses.append(response)
    finished = time.perf_counter()
    if not responses:
        raise RuntimeError("MLX LM returned no generation response")

    mx.eval([entry.state for entry in prompt_cache])
    final = responses[-1]
    output = "".join(response.text for response in responses)
    output_tokens = [int(response.token) for response in responses]
    from_draft = [bool(response.from_draft) for response in responses]
    target_cache = prompt_cache[:target_layer_count]
    draft_cache = prompt_cache[target_layer_count:]
    result = {
        "prompt_tokens": int(final.prompt_tokens),
        "output_token_ids": output_tokens,
        "generated_tokens": int(final.generation_tokens),
        "finish_reason": final.finish_reason,
        "eos_position_zero_based": next(
            (index for index, token in enumerate(output_tokens) if token in tokenizer.eos_token_ids),
            None,
        ),
        "generated_text": output,
        "generated_text_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "time_to_first_token_seconds": first_token_at - started,
        "request_seconds": finished - started,
        "decode_tokens_per_second": float(final.generation_tps),
        "mlx_active_memory_gb": mx.get_active_memory() / 1_000_000_000,
        "mlx_peak_memory_gb": mx.get_peak_memory() / 1_000_000_000,
        "process_peak_rss_gb": process_peak_rss_gb(),
        "target_cache": cache_summary(target_cache),
        "draft_cache": cache_summary(draft_cache) if draft_cache else None,
        "from_draft_flags": from_draft,
    }
    if draft_count is not None:
        result["draft_metrics"] = draft_cycle_metrics(from_draft, draft_count, max_tokens)
    return result


def build_cases(tokenizer, fixture_payload: dict) -> list[dict]:
    cases = []
    for definition in fixture_payload["baseline_cases"]:
        fixture = make_fixture(
            tokenizer,
            definition["target_prompt_tokens"],
            definition["prompt_type"],
        )
        cases.append({
            "id": definition["id"],
            "messages": fixture["messages"],
            "token_ids": fixture["token_ids"],
            "max_tokens": definition["max_tokens"],
            "suppress_eos": False,
        })
    for definition in fixture_payload["long_cases"]:
        messages = [
            {"role": "system", "content": definition["system"]},
            {"role": "user", "content": definition["user"]},
        ]
        token_ids = list(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )
        cases.append({
            "id": definition["id"],
            "messages": messages,
            "token_ids": token_ids,
            "max_tokens": definition["max_tokens"],
            "suppress_eos": definition.get("suppress_eos", False),
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--draft-model-path", type=Path, default=DEFAULT_DRAFT_PATH)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft-counts", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    args = parser.parse_args()

    if args.prefill_step_size < 1:
        parser.error("--prefill-step-size must be positive")
    if not args.draft_counts or any(count < 1 for count in args.draft_counts):
        parser.error("--draft-counts must contain positive integers")

    target_path = args.target_model_path if args.target_model_path.is_absolute() else ROOT / args.target_model_path
    draft_path = args.draft_model_path if args.draft_model_path.is_absolute() else ROOT / args.draft_model_path
    fixture_path = args.fixtures if args.fixtures.is_absolute() else ROOT / args.fixtures
    target_path = target_path.resolve()
    draft_path = draft_path.resolve()
    fixture_path = fixture_path.resolve()
    for path, label in ((target_path, "target"), (draft_path, "draft")):
        project_relative_path(path)
        if not path.is_dir() or not list(path.glob("*.safetensors")):
            raise FileNotFoundError(f"{label} model is missing weights: {path}")

    target_tokenizer_path = target_path / "tokenizer.json"
    draft_tokenizer_path = draft_path / "tokenizer.json"
    tokenizer_hashes = {
        "target": file_sha256(target_tokenizer_path),
        "draft": file_sha256(draft_tokenizer_path),
    }
    if tokenizer_hashes["target"] != tokenizer_hashes["draft"]:
        raise ValueError("Target and draft tokenizer.json files are not byte-identical")
    structural_signatures = {
        "target": structural_signature(target_path / "config.json"),
        "draft": structural_signature(draft_path / "config.json"),
    }
    if structural_signatures["target"] != structural_signatures["draft"]:
        raise ValueError("Target and draft structural model configurations differ")

    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    import mlx.core as mx

    started_at = datetime.now(timezone.utc)
    target_load_started = time.perf_counter()
    model, tokenizer = load(str(target_path))
    target_load_seconds = time.perf_counter() - target_load_started
    active_after_target_load = mx.get_active_memory() / 1_000_000_000
    draft_load_started = time.perf_counter()
    draft_model, draft_tokenizer = load(str(draft_path))
    draft_load_seconds = time.perf_counter() - draft_load_started
    active_after_both_load = mx.get_active_memory() / 1_000_000_000

    if draft_tokenizer.vocab_size != tokenizer.vocab_size:
        raise ValueError("Target and draft vocabulary sizes differ")

    fixture_payload = json.loads(fixture_path.read_text())
    cases = build_cases(tokenizer, fixture_payload)
    sampler = make_sampler(temp=0.0)
    result = {
        "schema_version": 1,
        "experiment": "experiment-6-speculative-decoding-stage-0",
        "started_at_utc": started_at.isoformat(),
        "environment": {
            "mlx": package_version("mlx"),
            "mlx_lm": package_version("mlx-lm"),
        },
        "target": {
            "id": DEFAULT_MODEL_ID,
            "local_path": project_relative_path(target_path),
            "weight_bytes": sum(path.stat().st_size for path in target_path.glob("*.safetensors")),
            "config_sha256": file_sha256(target_path / "config.json"),
            "load_seconds": target_load_seconds,
        },
        "draft": {
            "id": draft_path.name,
            "local_path": project_relative_path(draft_path),
            "weight_bytes": sum(path.stat().st_size for path in draft_path.glob("*.safetensors")),
            "config_sha256": file_sha256(draft_path / "config.json"),
            "load_seconds": draft_load_seconds,
        },
        "compatibility": {
            "tokenizer_sha256": tokenizer_hashes,
            "tokenizer_byte_identical": True,
            "structural_signatures": structural_signatures,
            "structural_config_identical": True,
            "target_vocab_size": tokenizer.vocab_size,
            "draft_vocab_size": draft_tokenizer.vocab_size,
        },
        "settings": {
            "sampling": "greedy",
            "prefill_step_size": args.prefill_step_size,
            "draft_counts": args.draft_counts,
            "fresh_cache_per_generation": True,
            "long_cases_suppress_eos_until_token_cap": True,
        },
        "memory_after_load": {
            "target_only_active_gb": active_after_target_load,
            "target_and_draft_active_gb": active_after_both_load,
        },
        "fixture": {
            "path": project_relative_path(fixture_path),
            "sha256": file_sha256(fixture_path),
        },
        "cases": [],
    }

    for case in cases:
        token_bytes = json.dumps(case["token_ids"], separators=(",", ":")).encode()
        logit_bias = (
            {token: -1e9 for token in tokenizer.eos_token_ids}
            if case["suppress_eos"]
            else None
        )
        logits_processors = make_logits_processors(logit_bias=logit_bias)
        print(f"Control: {case['id']}", flush=True)
        control = collect_generation(
            model=model,
            tokenizer=tokenizer,
            mx=mx,
            sampler=sampler,
            stream_generate=stream_generate,
            make_prompt_cache=make_prompt_cache,
            token_ids=case["token_ids"],
            max_tokens=case["max_tokens"],
            prefill_step_size=args.prefill_step_size,
            logits_processors=logits_processors,
        )
        case_result = {
            "id": case["id"],
            "messages": case["messages"],
            "actual_prompt_tokens": len(case["token_ids"]),
            "prompt_token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "max_tokens": case["max_tokens"],
            "suppress_eos": case["suppress_eos"],
            "control": control,
            "candidates": [],
        }
        result["cases"].append(case_result)
        write_json(args.output, result)

        for draft_count in args.draft_counts:
            print(f"Draft {draft_count}: {case['id']}", flush=True)
            candidate = collect_generation(
                model=model,
                tokenizer=tokenizer,
                mx=mx,
                sampler=sampler,
                stream_generate=stream_generate,
                make_prompt_cache=make_prompt_cache,
                token_ids=case["token_ids"],
                max_tokens=case["max_tokens"],
                prefill_step_size=args.prefill_step_size,
                logits_processors=logits_processors,
                draft_model=draft_model,
                draft_count=draft_count,
            )
            difference = first_difference(
                control["output_token_ids"], candidate["output_token_ids"]
            )
            candidate["comparison"] = {
                "exact_token_match": difference is None,
                "first_difference": difference,
                "finish_reason_match": control["finish_reason"] == candidate["finish_reason"],
                "eos_position_match": control["eos_position_zero_based"] == candidate["eos_position_zero_based"],
                "generated_length_match": control["generated_tokens"] == candidate["generated_tokens"],
                "target_cache_offsets_consistent": candidate["target_cache"]["offsets_consistent"],
                "draft_cache_offsets_consistent": candidate["draft_cache"]["offsets_consistent"],
            }
            candidate["comparison"]["stage0_pass"] = all(
                candidate["comparison"][key]
                for key in (
                    "exact_token_match",
                    "finish_reason_match",
                    "eos_position_match",
                    "generated_length_match",
                    "target_cache_offsets_consistent",
                    "draft_cache_offsets_consistent",
                )
            )
            case_result["candidates"].append({"draft_count": draft_count, **candidate})
            write_json(args.output, result)

    result["summary"] = {}
    for draft_count in args.draft_counts:
        comparisons = [
            candidate["comparison"]
            for case in result["cases"]
            for candidate in case["candidates"]
            if candidate["draft_count"] == draft_count
        ]
        result["summary"][str(draft_count)] = {
            "case_count": len(comparisons),
            "exact_match_count": sum(item["exact_token_match"] for item in comparisons),
            "stage0_pass_count": sum(item["stage0_pass"] for item in comparisons),
            "passed": all(item["stage0_pass"] for item in comparisons),
        }
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, result)
    print(f"Saved Stage-0 result to {args.output}", flush=True)


if __name__ == "__main__":
    main()

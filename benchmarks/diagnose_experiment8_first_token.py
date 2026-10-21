#!/usr/bin/env python3
"""Reproduce Experiment 8's first-token log-probability margin; no speed result."""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path

from run_baseline import ROOT, DEFAULT_MODEL_PATH, file_sha256, make_fixture, package_version, write_json
from run_experiment8_packed import DATE, capacity_check, digest, machine_state, now, sync

CASE_ID = "natural-64-context_question"


def read_reference(path):
    """Read original JSON bytes from an explicit gzip path or archived fallback."""
    path = Path(path)
    if not path.is_file() and path.suffix != ".gz":
        archived = path.with_name(path.name + ".gz")
        if archived.is_file():
            path = archived
    raw = path.read_bytes()
    logical = gzip.decompress(raw) if path.suffix == ".gz" else raw
    payload = json.loads(logical)
    if not isinstance(payload, dict):
        raise ValueError("reference JSON payload must be an object")
    return path, logical, payload


def first_response(model, tokenizer, mx, tokens):
    """Keep the original cap/lookahead schedule, inspect only its first response."""
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    cache = make_prompt_cache(model)
    sync(mx)
    iterator = stream_generate(
        model, tokenizer, tokens, max_tokens=128, sampler=make_sampler(temp=0),
        prompt_cache=cache, prefill_step_size=2048,
    )
    try:
        response = next(iterator)
    finally:
        iterator.close()
        sync(mx)
    logprobs = response.logprobs
    if logprobs.ndim != 1:
        raise AssertionError("expected one vocabulary log-probability vector")
    indices = mx.argpartition(logprobs, kth=-2)[-2:]
    values = logprobs[indices]
    reference_ids = [40, 11439]
    reference_values = logprobs[mx.array(reference_ids)]
    mx.eval(indices, values, reference_values)
    ranked = sorted(
        [{"token_id": int(token), "log_probability": float(value),
          "decoded_label": tokenizer.decode([int(token)])}
         for token, value in zip(indices.tolist(), values.tolist())],
        key=lambda item: (-item["log_probability"], item["token_id"]),
    )
    margin = ranked[0]["log_probability"] - ranked[1]["log_probability"]
    if not math.isfinite(margin):
        raise AssertionError("nonfinite first-token top-two margin")
    offsets = [int(entry.offset) for entry in cache]
    expected_offset = len(tokens) + 1
    if len(offsets) != 16 or set(offsets) != {expected_offset}:
        raise AssertionError("first-response lookahead cache offset mismatch")
    selected = int(response.token)
    record = {
        "observed_first_token_id": selected,
        "observed_first_token_label": tokenizer.decode([selected]),
        "observed_first_response_text": response.text,
        "top_two": ranked,
        "top_two_log_probability_margin": margin,
        "declared_reference_token_log_probabilities": [
            {"token_id": token, "decoded_label": tokenizer.decode([token]),
             "log_probability": float(value)}
            for token, value in zip(reference_ids, reference_values.tolist())
        ],
        "logprob_dtype": str(logprobs.dtype),
        "cache_offsets_after_original_lookahead": offsets,
        "expected_cache_offset": expected_offset,
        "request_cap": 128,
        "responses_consumed": 1,
        "note": "Top two use argpartition; tied candidate membership need not establish the sampler tie-break. Values are exposed log probabilities, not unrounded logits.",
    }
    del response, iterator, logprobs, indices, values, reference_values, cache
    sync(mx)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=ROOT/"benchmarks/results/experiment8-packed-C-correctness-20261020-r3.json")
    parser.add_argument("--candidate", type=Path, default=ROOT/"benchmarks/results/experiment8-packed-Q-correctness-20261020-r3.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.control, args.candidate, args.output = (p.resolve() for p in (args.control, args.candidate, args.output))
    for path in (args.control, args.candidate, args.output):
        path.relative_to(ROOT)
    partial = args.output.with_suffix(".partial.json")
    if args.output.exists() or partial.exists():
        parser.error("refusing to overwrite diagnostic output")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    reference_data = [read_reference(path) for path in (args.control, args.candidate)]
    args.control, args.candidate = (entry[0] for entry in reference_data)
    refs = [entry[2] for entry in reference_data]
    for ref, arm in zip(refs, ("C", "Q")):
        if ref.get("arm") != arm or ref.get("stage") != "correctness":
            raise ValueError("wrong reference arm/stage")
    if refs[0].get("status") != "pass" or refs[1].get("status") != "rejected":
        raise ValueError("requires completed C reference and rejected Q reference")
    if refs[0]["manifest"] != refs[1]["manifest"]:
        raise ValueError("reference model/source manifests differ")
    for ref, golden in zip(refs, (40, 11439)):
        if ref["checks"][CASE_ID]["generation"]["output_token_ids"][0] != golden:
            raise ValueError("reference first token differs from declared diagnostic")
    result = {
        "schema_version": 1, "experiment": "experiment8-first-token-margin-diagnostic",
        "started_at_utc": now(), "status": "running", "case_id": CASE_ID,
        "references": {str(path.relative_to(ROOT)): hashlib.sha256(logical).hexdigest()
                       for path, logical, _ in reference_data},
        "reference_container_sha256": {str(path.relative_to(ROOT)): file_sha256(path)
                                       for path, _, _ in reference_data},
        "reference_hash_semantics": "references hashes original decompressed JSON bytes; container hashes identify retained plain/gzip files.",
        "reference_manifest": refs[0]["manifest"],
        "existing_source_snapshot_hashes": {str(path.relative_to(ROOT)): file_sha256(path)
                                           for path in sorted((ROOT/"benchmarks/results").glob("experiment8-source-*")) if path.is_file()},
        "source_manifest": {str(Path(__file__).relative_to(ROOT)): file_sha256(Path(__file__)),
                            "benchmarks/experiment8_packing.py": file_sha256(ROOT/"benchmarks/experiment8_packing.py"),
                            "benchmarks/run_baseline.py": file_sha256(ROOT/"benchmarks/run_baseline.py"),
                            "benchmarks/run_experiment8_packed.py": file_sha256(ROOT/"benchmarks/run_experiment8_packed.py")},
        "settings": {"date": DATE, "prefill_step_size": 2048, "greedy": True,
                     "fresh_cache_per_arm": True, "max_tokens": 128, "consumed_responses": 1},
        "interpretation": "Diagnostic only: no full generation, latency result, gate relaxation, or selected deployment change.",
    }
    write_json(partial, result)
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx.utils import tree_flatten
        from experiment8_packing import pack_model

        if (package_version("mlx"), package_version("mlx-lm")) != ("0.32.2", "0.31.3"):
            raise RuntimeError("runtime_incompatible")
        if mx.device_info()["device_name"] != "Apple M2 Pro":
            raise RuntimeError("device_incompatible")
        result["state_start"] = machine_state()
        for name in ("mlx_lm.models.llama", "mlx_lm.generate", "mlx.nn.layers.quantized"):
            result["source_manifest"][name] = file_sha256(Path(importlib.import_module(name).__file__))
        for key, expected in refs[0]["manifest"].items():
            if key.startswith("experiments/"):
                continue
            current = result["source_manifest"].get(key)
            if current is None:
                path = ROOT/key
                if not path.is_file():
                    raise ValueError("reference input unavailable: " + key)
                current = file_sha256(path)
                result["source_manifest"][key] = current
            if current != expected:
                raise ValueError("runtime/model/source differs from rejected capture: " + key)
        model, tokenizer = load(str(DEFAULT_MODEL_PATH))
        mx.eval(model.parameters())
        sync(mx)
        fixture = make_fixture(tokenizer, 64, "context_question", DATE)
        tokens = fixture["token_ids"]
        result["fixture"] = {"prompt_token_ids": tokens, "prompt_sha256": digest(tokens),
                             "messages": fixture["messages"], "actual_prompt_tokens": len(tokens)}
        for ref in refs:
            if digest(tokens) != ref["checks"][CASE_ID]["prompt_sha256"]:
                raise ValueError("fixture differs from reference")
        before_bytes = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
        result["C"] = first_response(model, tokenizer, mx, tokens)
        if result["C"]["observed_first_token_id"] != 40:
            raise AssertionError("control_first_token_not_reproduced")
        result["state_after_control"] = capacity_check(result["state_start"])
        write_json(partial, result)
        gc.collect()
        mx.clear_cache()
        result["packing"] = pack_model(model, "Q", verify=None)
        sync(mx)
        after_bytes = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
        result["parameter_bytes"] = {"before": before_bytes, "after": after_bytes}
        if before_bytes != after_bytes:
            raise AssertionError("parameter_bytes_changed")
        result["Q"] = first_response(model, tokenizer, mx, tokens)
        result["candidate_first_token_reproduced"] = result["Q"]["observed_first_token_id"] == 11439
        result["state_end"] = capacity_check(result["state_start"])
        result["status"] = "pass" if result["candidate_first_token_reproduced"] else "not_reproduced"
        result["completed_at_utc"] = now()
    except Exception as error:
        result["status"] = "stopped"
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        result["terminated_at_utc"] = now()
    write_json(args.output, result)
    partial.unlink(missing_ok=True)
    print(json.dumps({"status": result["status"], "output": str(args.output.relative_to(ROOT)),
                      "failure": result.get("failure")}), flush=True)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

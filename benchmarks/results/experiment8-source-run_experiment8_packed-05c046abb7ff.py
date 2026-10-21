#!/usr/bin/env python3
"""Experiment 8: isolated-process correctness and timing worker (offline only)."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from run_baseline import (ROOT, DEFAULT_MODEL_PATH, DEFAULT_TARGETS, PROMPT_TYPES,
                          file_sha256, make_fixture, package_version, summarize_runs, write_json)

DATE = "04 Oct 2026"
SHAPES = (1, 2, 9, 10, 11, 12, 13, 17, 18, 19, 31, 32, 33, 63, 64, 65,
          2046, 2047, 2048, 2049, 4095)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def machine_state():
    swap = command("sysctl", "vm.swapusage")
    match = re.search(r"used = ([0-9.]+)M", swap or "")
    power = command("pmset", "-g", "custom")
    batt = command("pmset", "-g", "batt")
    rss = command("ps", "-o", "rss=", "-p", str(os.getpid()))
    return {
        "swap_used_bytes": int(float(match[1]) * 2**20) if match else None,
        "power_config": power,
        "supply": (batt or "").splitlines()[0] if batt else None,
        "thermal": command("pmset", "-g", "therm"),
        "process_rss_bytes": int(rss.strip()) * 1024 if rss and rss.strip().isdigit() else None,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def capacity_check(start):
    state = machine_state()
    if state["process_rss_bytes"] is None or start["process_rss_bytes"] is None:
        raise RuntimeError("capacity_unmeasured: current RSS counter unavailable")
    if state["process_peak_rss_bytes"] > 8 * 2**30:
        raise RuntimeError("capacity_stop: process RSS high-water exceeds 8 GiB")
    if start["swap_used_bytes"] is None or state["swap_used_bytes"] is None:
        raise RuntimeError("capacity_unmeasured: swap counter unavailable")
    if state["swap_used_bytes"] > start["swap_used_bytes"]:
        raise RuntimeError("capacity_stop: system swap grew during arm")
    if (state["power_config"], state["supply"]) != (start["power_config"], start["supply"]):
        raise RuntimeError("power_state_changed")
    return state


def memory(mx):
    return {"active_bytes": mx.get_active_memory(), "cached_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def sync(mx):
    from mlx_lm.generate import generation_stream
    mx.synchronize(generation_stream)
    mx.synchronize()


def cache_signature(cache):
    return [{"type": type(c).__name__, "offset": int(c.offset),
             "shapes": [list(a.shape) for a in c.state], "bytes": c.nbytes} for c in cache]


def quality(text, kind, expected=None):
    if expected is not None:
        from run_experiment2 import normalize_fact_text
        # Declared singular/plural alternative; token boundaries avoid partial-word passes.
        alternatives = {"swollen batteries": ("swollen batteries", "swollen battery")}.get(expected, (expected,))
        normalized = normalize_fact_text(text)
        matched = any(re.search(r"(?<!\w)" + re.escape(normalize_fact_text(fact)) + r"(?!\w)", normalized)
                      for fact in alternatives)
        return {"required_fact": matched, "expected": expected, "alternatives": list(alternatives)}
    if kind == "summarization":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return {"exactly_three_bullets": len(lines) == 3 and
                all(re.match(r"^[-*•]\s+\S", line) is not None for line in lines)}
    if kind == "extraction":
        value = text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", value, re.DOTALL)
        if fenced:
            value = fenced[1]
        try:
            parsed = json.loads(value)
            valid = isinstance(parsed, dict) and bool(parsed)
        except (ValueError, TypeError):
            valid = False
        return {"complete_json_object": valid,
                "semantic_schema": "unmeasured: baseline wording declares no fixed keys"}
    return {"task_semantics": "unmeasured; token equivalence assessed separately"}


def generation(model, tokenizer, mx, tokens, cap=128, forced=False, cache=None, validate_cache=False):
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    cache = cache if cache is not None else make_prompt_cache(model)
    initial_offset = int(cache[0].offset)
    sync(mx)
    mx.reset_peak_memory()
    processors = make_logits_processors(logit_bias={t: -1e9 for t in tokenizer.eos_token_ids}) if forced else []
    ids, parts, margins = [], [], []
    final_reason = None
    reported_count = None
    first = None
    iterator = stream_generate(model, tokenizer, tokens, max_tokens=cap, sampler=make_sampler(temp=0),
                               prompt_cache=cache, prefill_step_size=2048,
                               logits_processors=processors)
    started = time.perf_counter()
    try:
        for response in iterator:
            if first is None:
                first = time.perf_counter()
            ids.append(int(response.token))
            parts.append(response.text)
            # Copy only scalar evidence, never retain vocabulary-sized response arrays.
            # Margin collection is outside primary timing to avoid extra GPU work.
            final_reason = response.finish_reason
            reported_count = int(response.generation_tokens)
            prompt_tps = float(response.prompt_tps)
            api_decode_tps = float(response.generation_tps)
        sync(mx)
        finished = time.perf_counter()
        peaks = memory(mx)
    finally:
        iterator.close()
    if first is None:
        raise RuntimeError("empty generation")
    del response, iterator
    state = cache_signature(cache)
    offsets = {c["offset"] for c in state}
    if len(offsets) != 1 or len(state) != 16:
        raise AssertionError("invalid cache layer/offset consistency")
    expected_offset = initial_offset + len(tokens) + len(ids)
    if offsets != {expected_offset}:
        raise AssertionError(f"cache_offset_mismatch: {offsets} versus {expected_offset}")
    eos = [i for i, token in enumerate(ids) if token in tokenizer.eos_token_ids]
    if forced and (len(ids) != cap or eos or final_reason != "length"):
        raise AssertionError("forced_output_contract_failed")
    cache_finite = None
    if validate_cache:
        finite_checks = [mx.all(mx.isfinite(a)) for c in cache for a in c.state]
        mx.eval(finite_checks)
        cache_finite = all(bool(a.item()) for a in finite_checks)
        if not cache_finite:
            raise AssertionError("nonfinite_cache_state")
    text = "".join(parts)
    result = {"output_token_ids": ids, "generated_tokens": reported_count,
              "response_token_count": len(ids), "visible_non_eos_count": len(ids) - len(eos),
              "eos_position_zero_based": eos[0] if eos else None, "finish_reason": final_reason,
              "generated_text": text, "generated_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "time_to_first_token_seconds": first - started, "request_seconds": finished - started,
              "decode_seconds": finished - first,
              "decode_tokens_per_second": max(0, len(ids) - 1) / (finished - first),
              "api_decode_tokens_per_second": api_decode_tps,
              "prefill_tokens_per_second": prompt_tps,
              "mlx_peak_memory_gb": peaks["peak_bytes"] / 1e9,
              "mlx_active_memory_gb": peaks["active_bytes"] / 1e9,
              "process_peak_rss_gb": peaks["process_peak_rss_bytes"] / 1e9,
              "memory": peaks, "cache": state, "expected_cache_offset": expected_offset,
              "cache_finite": cache_finite, "cached_token_history_sha256": digest(tokens + ids),
              "token_margin": "unmeasured; no extra GPU reduction inside primary timing"}
    return result, cache


def compare(actual, expected):
    keys = ("output_token_ids", "generated_tokens", "response_token_count", "eos_position_zero_based",
            "finish_reason", "cache", "generated_text_sha256")
    failures = [k for k in keys if actual[k] != expected[k]]
    difference = None
    if "output_token_ids" in failures:
        a, b = actual["output_token_ids"], expected["output_token_ids"]
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        difference = {"index": i, "control_token": b[i] if i < len(b) else None,
                      "candidate_token": a[i] if i < len(a) else None}
    return {"passed": not failures, "different_fields": failures, "first_token_difference": difference}


def capture_activations(model, tokenizer, mx):
    from mlx_lm.models.base import create_attention_mask
    fixture = make_fixture(tokenizer, 2048, "summarization", DATE)
    h = model.model.embed_tokens(mx.array([fixture["token_ids"]]))
    mask = create_attention_mask(h, None)
    saved = {}
    for i, layer in enumerate(model.layers):
        q = layer.input_layernorm(h)
        qsmall = mx.take(q, mx.arange(min(65, q.shape[1])), axis=1)
        mx.eval(qsmall)
        h = h + layer.self_attn(q, mask, cache=None)
        g = layer.post_attention_layernorm(h)
        gsmall = mx.take(g, mx.arange(min(65, g.shape[1])), axis=1)
        mx.eval(gsmall)
        h = h + layer.mlp(g)
        mx.eval(h)
        saved[i] = {"Q": qsmall, "G": gsmall}
    del h, q, g
    sync(mx)
    return saved


def numeric_verifier(mx, activations, progress, state_start):
    def verify(i, family, originals, packed):
        evidence = []
        progress("packed_bytes_verified", {"layer": i, "family": family, "packing": packed.check})
        dtype = activations[i][family].dtype
        for rows in SHAPES:
            for kind in ("zeros", "ones", "random", "activation"):
                shape = (1, rows, 2048)
                if kind == "zeros":
                    x = mx.zeros(shape, dtype=dtype)
                elif kind == "ones":
                    x = mx.ones(shape, dtype=dtype)
                elif kind == "random":
                    mx.random.seed(20261020 + i * 10000 + rows)
                    x = mx.random.uniform(-1, 1, shape, dtype=dtype)
                else:
                    a = activations[i][family]
                    x = mx.tile(a, (1, math.ceil(rows/a.shape[1]), 1))[:, :rows, :]
                mx.eval(x)
                parts = packed.split(x)
                for j, (original, candidate) in enumerate(zip(originals, parts)):
                    separate = original(x)
                    delta = separate.astype(mx.float32) - candidate.astype(mx.float32)
                    finite = mx.all(mx.isfinite(candidate)) & mx.all(mx.isfinite(separate))
                    good = mx.allclose(candidate, separate, atol=1e-2, rtol=1e-2)
                    maximum, rms = mx.max(mx.abs(delta)), mx.sqrt(mx.mean(delta * delta))
                    mx.eval(finite, good, maximum, rms)
                    row = {"layer": i, "family": family, "rows": rows, "input": kind,
                           "dtype": str(dtype), "component": j,
                           "max_abs": float(maximum.item()), "rms": float(rms.item()),
                           "finite": bool(finite.item()), "allclose": bool(good.item()),
                           "output_shape": list(candidate.shape),
                           "output_strides": list(candidate.strides) if hasattr(candidate, "strides") else None}
                    evidence.append(row)
                    if not row["finite"] or not row["allclose"]:
                        progress("numerical_failure", row)
                        raise AssertionError("numerical_failure: " + json.dumps(row))
                    del separate, delta, finite, good, maximum, rms
                del x, parts, candidate
            capacity_check(state_start)
        progress("numeric_family_pass", {"layer": i, "family": family, "checks": len(evidence)})
        return {"passed": True, "checks": evidence}
    return verify


def build_cases(tokenizer, expected=None):
    cases = []
    for target in DEFAULT_TARGETS:
        for kind in PROMPT_TYPES:
            f = make_fixture(tokenizer, target, kind, DATE)
            cases.append({"id": f"natural-{target}-{kind}", "tokens": f["token_ids"],
                          "kind": kind, "target": target, "cap": 128, "forced": False})
        f = make_fixture(tokenizer, target, "summarization", DATE)
        cases.append({"id": f"forced-{target}-128", "tokens": f["token_ids"], "kind": "forced",
                      "target": target, "cap": 128, "forced": True})
    f = make_fixture(tokenizer, 2048, "summarization", DATE)
    cases.append({"id": "forced-2048-256", "tokens": f["token_ids"], "kind": "forced",
                  "target": 2048, "cap": 256, "forced": True})
    return cases


def histories(model, tokenizer, mx, result, expected, save):
    from mlx_lm.models.cache import make_prompt_cache
    source = json.loads((ROOT/"benchmarks/fixtures/experiment2-sessions.json").read_text())
    for scenario in source["scenarios"]:
        # Same Experiment 2 construction with fixed template date.
        notes = list(scenario["notes"])
        n = 0
        def render(messages):
            return list(tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, *messages], tokenize=True,
                add_generation_prompt=True, date_string=DATE))
        while True:
            system = f"Reference file: {scenario['title']}. Answer from these notes.\n" + "\n".join("- " + x for x in notes)
            if len(render([{"role": "user", "content": scenario["turns"][0]["user"]}])) >= scenario["profile_tokens"]:
                break
            notes.append(f"Supplemental record {n+1}: {scenario['notes'][n % len(scenario['notes'])]}")
            n += 1
        messages = []
        # Freeze a control fresh-policy transcript. Reuse and candidates consume these exact histories.
        for turn, definition in enumerate(scenario["turns"]):
            messages.append({"role": "user", "content": definition["user"]})
            tokens = render(messages)
            key = f"history-{scenario['id']}-fresh-{turn}"
            if expected:
                transcript = expected["checks"][key]["generation"]["generated_text"]
            else:
                record, cache = generation(model, tokenizer, mx, tokens, cap=128, validate_cache=True)
                del cache
                result["checks"][key] = {"prompt_sha256": digest(tokens), "prompt_token_ids": tokens,
                    "generation": record, "quality": quality(record["generated_text"], "history", definition["expected_fact"])}
                transcript = record["generated_text"]
                result["state_last"] = capacity_check(result["state_start"])
                save()
            messages.append({"role": "assistant", "content": transcript})
        for policy in ("fresh", "reuse"):
            cache, represented = None, []
            for turn, definition in enumerate(scenario["turns"]):
                key = f"history-{scenario['id']}-{policy}-{turn}"
                frozen = result["checks"].get(f"history-{scenario['id']}-fresh-{turn}") if expected is None else expected["checks"][f"history-{scenario['id']}-fresh-{turn}"]
                tokens = frozen["prompt_token_ids"]
                if policy == "fresh" and expected is None:
                    continue
                reused = 0
                if policy == "reuse" and cache is not None:
                    reused = next((k for k, (a,b) in enumerate(zip(represented,tokens)) if a != b), min(len(represented),len(tokens)))
                    reused = min(reused, len(tokens)-1, min(c.offset for c in cache))
                    for c in cache:
                        c.trim(c.offset - reused)
                else:
                    cache = make_prompt_cache(model)
                record, cache = generation(model, tokenizer, mx, tokens[reused:], cache=cache, validate_cache=True)
                represented = tokens + record["output_token_ids"]
                represented = represented[:min(c.offset for c in cache)]
                entry = {"prompt_sha256": digest(tokens), "prompt_token_ids": tokens,
                         "reused_tokens": reused, "generation": record,
                         "quality": quality(record["generated_text"], "history", definition["expected_fact"])}
                result["checks"][key] = entry
                if expected:
                    entry["comparison"] = compare(record, expected["checks"][key]["generation"])
                    if entry["prompt_sha256"] != expected["checks"][key]["prompt_sha256"] or not entry["comparison"]["passed"]:
                        save()
                        raise AssertionError("history_mismatch: " + key)
                result["state_last"] = capacity_check(result["state_start"])
                save()
            del cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("C", "Q", "G", "QG"), required=True)
    parser.add_argument("--stage", choices=("correctness", "screen", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--reverse-fixtures", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or args.output.with_suffix(".partial.json").exists():
        parser.error("refusing to overwrite result")
    if args.arm != "C" and args.control is None:
        parser.error("candidate requires matched control")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from mlx_lm import load
    from mlx.utils import tree_flatten
    expected = json.loads(args.control.read_text()) if args.control else None
    if expected and (not expected.get("completed_at_utc") or expected.get("status") != "pass"
                     or expected.get("arm") != "C" or expected.get("stage") != args.stage):
        raise ValueError("control is incomplete, failed, or has wrong arm/stage")
    state_start = machine_state()
    result = {"schema_version": 1, "experiment": "experiment8-packed-projections", "arm": args.arm,
              "stage": args.stage, "started_at_utc": now(), "status": "running",
              "workload_id": "exp8-packed-" + {"correctness": "correctness-v1", "screen": "screen-v1", "full": "diverse-v1"}[args.stage],
              "environment": {"python": platform.python_version(), "os": platform.platform(),
                              "mlx": package_version("mlx"), "mlx_lm": package_version("mlx-lm"),
                              "device": mx.device_info()}, "state_start": state_start,
              "settings": {"date": DATE, "prefill_step_size": 2048, "greedy": True,
                           "fresh_primary_cache": True, "concurrency": 1,
                           "numerical_shapes": SHAPES, "numerical_seed": 20261020},
              "checks": {}, "cases": [], "events": [], "manifest": {}}
    partial = args.output.with_suffix(".partial.json")
    def save():
        payload_bytes = len(json.dumps(result, ensure_ascii=False).encode())
        retained = sum(p.stat().st_size for p in (ROOT/"benchmarks/results").glob("experiment8*") if p.is_file() and p != partial)
        if retained + payload_bytes * 2 > 10_000_000:
            raise RuntimeError("storage_budget_stop: Experiment 8 outputs would exceed 10 MB")
        write_json(partial, result)
    def progress(kind, data):
        result["events"].append({"time": now(), "kind": kind, "data": data})
        save()
        print(json.dumps({"kind": kind, "data": data}), flush=True)
    save()
    try:
        if (result["environment"]["mlx"], result["environment"]["mlx_lm"]) != ("0.32.2", "0.31.3"):
            raise RuntimeError("runtime_incompatible")
        if result["environment"]["device"]["device_name"] != "Apple M2 Pro":
            raise RuntimeError("device_incompatible")
        paths = list(DEFAULT_MODEL_PATH.glob("*.safetensors")) + [DEFAULT_MODEL_PATH/n for n in ("config.json", "tokenizer.json", "tokenizer_config.json")]
        paths += [ROOT/"experiments"/f"experiment-8-packed-projections-{name}.md" for name in ("plan","spec")]
        paths += [Path(__file__), ROOT/"benchmarks/run_baseline.py", ROOT/"benchmarks/experiment8_packing.py", ROOT/"benchmarks/fixtures/experiment2-sessions.json"]
        for module in ("mlx_lm.models.llama", "mlx_lm.generate", "mlx.nn.layers.quantized"):
            path = Path(importlib.import_module(module).__file__)
            result["manifest"][module] = file_sha256(path)
        for path in paths:
            result["manifest"][str(path.relative_to(ROOT))] = file_sha256(path)
        if expected:
            for key in ("workload_id", "settings", "manifest", "environment"):
                if result[key] != expected[key]:
                    raise ValueError(f"matched_control_contract_mismatch: {key}")
        result["state_last"] = capacity_check(state_start)
        result["frozen_sources"] = {}
        for path in paths:
            if path.suffix not in (".py", ".md"):
                continue
            checksum = file_sha256(path)
            snapshot = ROOT/"benchmarks/results"/f"experiment8-source-{path.stem}-{checksum[:12]}{path.suffix}"
            if snapshot.exists() and file_sha256(snapshot) != checksum:
                raise ValueError("source_snapshot_collision")
            if not snapshot.exists():
                snapshot.write_bytes(path.read_bytes())
            result["frozen_sources"][str(path.relative_to(ROOT))] = str(snapshot.relative_to(ROOT))
        result["memory_preload"] = memory(mx)
        mx.reset_peak_memory()
        started = time.perf_counter()
        model, tokenizer = load(str(DEFAULT_MODEL_PATH))
        mx.eval(model.parameters())
        sync(mx)
        result["load_seconds"] = time.perf_counter() - started
        result["memory_loaded"] = memory(mx)
        result["state_last"] = capacity_check(state_start)
        before_bytes = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
        result["parameter_bytes_before"] = before_bytes
        if not model.args.tie_word_embeddings or hasattr(model, "lm_head"):
            raise AssertionError("tied_head_incompatible")
        if args.arm != "C":
            from experiment8_packing import pack_model
            activations = capture_activations(model, tokenizer, mx) if args.stage == "correctness" else None
            mx.clear_cache()
            mx.reset_peak_memory()
            pack_started = time.perf_counter()
            result["packing"] = pack_model(model, args.arm,
                verify=numeric_verifier(mx, activations, progress, state_start) if activations else None)
            sync(mx)
            result["packing_seconds"] = time.perf_counter() - pack_started
            result["memory_packing"] = memory(mx)
            # Diagnostic matrices are included in correctness-process peak only.
            # Clean timing workers independently measure packing-only peak.
            del activations
        after_bytes = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
        result["parameter_bytes_after"] = after_bytes
        if before_bytes != after_bytes:
            raise AssertionError("parameter_bytes_changed")
        gc.collect()
        mx.clear_cache()
        result["memory_settled"] = memory(mx)
        if args.stage != "correctness" and expected:
            peak = result.get("memory_packing", result["memory_loaded"])["peak_bytes"]
            reference = expected["memory_loaded"]["peak_bytes"]
            if peak - reference > 512 * 2**20:
                raise AssertionError("packing_peak_gate_failed")
        cases = build_cases(tokenizer)
        if args.stage == "correctness":
            for case in cases:
                record, cache = generation(model, tokenizer, mx, case["tokens"], case["cap"], case["forced"], validate_cache=True)
                del cache
                entry = {"prompt_sha256": digest(case["tokens"]), "generation": record,
                         "quality": quality(record["generated_text"], case["kind"]) if not case["forced"] else {"forced_output": True}}
                result["checks"][case["id"]] = entry
                if expected:
                    entry["comparison"] = compare(record, expected["checks"][case["id"]]["generation"])
                    if entry["prompt_sha256"] != expected["checks"][case["id"]]["prompt_sha256"] or not entry["comparison"]["passed"]:
                        save()
                        raise AssertionError("whole_model_mismatch: " + case["id"])
                result["state_last"] = capacity_check(state_start)
                progress("whole_model_pass", {"id": case["id"], "tokens": len(record["output_token_ids"])})
            histories(model, tokenizer, mx, result, expected, save)
        else:
            cold_fixture = make_fixture(tokenizer, 64, "qa", DATE)
            cold, cold_cache = generation(model, tokenizer, mx, cold_fixture["token_ids"])
            result["cold_first_request"] = cold
            del cold_cache
            cases = [c for c in cases if c["id"].startswith("forced-") and c["cap"] == 128] if args.stage == "screen" else [c for c in cases if not c["forced"]]
            if args.reverse_fixtures:
                cases.reverse()
            for case in cases:
                record, cache = generation(model, tokenizer, mx, case["tokens"], case["cap"], case["forced"])
                del cache
                item = {"id": case["id"], "target_prompt_tokens": case["target"],
                        "actual_prompt_tokens": len(case["tokens"]), "prompt_sha256": digest(case["tokens"]),
                        "warmup": record, "runs": []}
                result["cases"].append(item)
                control_case = next((c for c in expected["cases"] if c["id"] == case["id"]), None) if expected else None
                if expected:
                    if control_case is None:
                        raise ValueError("matched control is missing timing case")
                    if item["prompt_sha256"] != control_case["prompt_sha256"] or not compare(item["warmup"], control_case["warmup"])["passed"]:
                        raise AssertionError("warmup_correctness_mismatch: " + case["id"])
                for rep in range(5 if args.stage == "screen" else 10):
                    record, cache = generation(model, tokenizer, mx, case["tokens"], case["cap"], case["forced"])
                    del cache
                    item["runs"].append(record)
                    if control_case and (item["prompt_sha256"] != control_case["prompt_sha256"] or not compare(record, control_case["runs"][rep])["passed"]):
                        raise AssertionError("timing_correctness_mismatch: " + case["id"])
                    result["state_last"] = capacity_check(state_start)
                    save()
                item["summary"] = summarize_runs(item["runs"])
                progress("timing_case_complete", {"id": case["id"]})
        quality_checks = {k: v["quality"] for k,v in result["checks"].items()}
        result["quality_evidence"] = quality_checks
        result["absolute_quality_pass"] = (all(
            flag is not False for q in quality_checks.values() for flag in q.values())
            if quality_checks else None)
        result["absolute_quality_scope"] = "Declared structural/fact checks only; no extraction schema or semantic evaluation is claimed."
        result["state_last"] = capacity_check(state_start)
        result["status"] = "pass"
        result["completed_at_utc"] = now()
    except Exception as error:
        result["status"] = "rejected" if isinstance(error, (AssertionError, ValueError)) else "stopped"
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        result["terminated_at_utc"] = now()
        result["later_stages"] = "not run after hard gate failure"
    result["state_end"] = machine_state()
    if result["status"] == "pass":
        try:
            capacity_check(state_start)
        except RuntimeError as error:
            result["status"] = "stopped"
            result["failure"] = {"type": type(error).__name__, "message": str(error)}
            result.pop("completed_at_utc", None)
            result["terminated_at_utc"] = now()
    write_json(args.output, result)
    if partial.exists():
        partial.unlink()
    print(json.dumps({"status": result["status"], "failure": result.get("failure"), "output": str(args.output.relative_to(ROOT))}), flush=True)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

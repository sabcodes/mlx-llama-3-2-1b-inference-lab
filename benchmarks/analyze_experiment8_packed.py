#!/usr/bin/env python3
"""Offline raw-evidence gates for Experiment 8; never imports MLX or runs inference.

API: analyze_screen(payloads, correctness=None), analyze_pair(control, candidate,
correctness=None), analyze_confirmation(pairs, selected_arm, correctness=None).
Correctness is an arm -> completed Stage-0 payload mapping, required to select.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import gzip
import json
import math
from pathlib import Path
import statistics
from functools import lru_cache

TARGETS = (64, 512, 2048, 4096)
KINDS = ("qa", "rewrite", "context_question", "extraction", "summarization")
ARMS = ("C", "Q", "G", "QG")
ROWS = (1, 2, 9, 10, 11, 12, 13, 17, 18, 19, 31, 32, 33, 63, 64, 65,
        2046, 2047, 2048, 2049, 4095)
IDENTITY = ("output_token_ids", "generated_tokens", "response_token_count",
            "visible_non_eos_count", "eos_position_zero_based", "finish_reason",
            "generated_text_sha256", "cache", "expected_cache_offset")
METRICS = ("request_seconds", "time_to_first_token_seconds", "mlx_peak_memory_gb")


class EvidenceError(ValueError):
    pass


def require(value, message):
    if not value:
        raise EvidenceError(message)


def number(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), f"invalid {label}")
    return value


def sha(value, label):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), f"invalid {label} hash")


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def stamp(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(result.tzinfo is not None, "timestamp has no timezone")
        return result
    except (TypeError, ValueError, AttributeError) as error:
        raise EvidenceError("invalid completion/start timestamp") from error


def nearest_rank(values, probability=.95):
    require(bool(values), "empty quantile input")
    return sorted(values)[math.ceil(probability * len(values)) - 1]


def percent(candidate, control):
    require(control > 0, "zero metric denominator")
    return (candidate / control - 1) * 100


def allowed(value, limit):
    # Numerical epsilon only protects decimal roundoff exactly on a frozen gate.
    return value <= limit + 1e-9


def expected_case_ids(stage):
    if stage == "screen":
        return {f"forced-{n}-128" for n in TARGETS}
    return {f"natural-{n}-{kind}" for n in TARGETS for kind in KINDS}


@lru_cache(maxsize=1)
def history_definitions():
    source = Path(__file__).resolve().parents[1] / "benchmarks/fixtures/experiment2-sessions.json"
    sessions = json.loads(source.read_text())["scenarios"]
    return {f"history-{s['id']}-{policy}-{i}": turn for s in sessions
            for policy in ("fresh", "reuse") for i, turn in enumerate(s["turns"])}


def evaluate_quality_check(key, text):
    # This import loads only stdlib helpers; the worker imports MLX inside main.
    # Reuse the frozen validator rather than silently changing its interpretation.
    from run_experiment8_packed import quality
    if key.startswith("forced-"):
        return {"forced_output": True}
    if key.startswith("history-"):
        return quality(text, "history", history_definitions()[key]["expected_fact"])
    return quality(text, key.split("-", 2)[2])


def expected_correctness_ids():
    ids = expected_case_ids("full") | expected_case_ids("screen") | {"forced-2048-256"}
    source = Path(__file__).resolve().parents[1] / "benchmarks/fixtures/experiment2-sessions.json"
    sessions = json.loads(source.read_text())["scenarios"]
    ids |= {f"history-{s['id']}-{policy}-{i}" for s in sessions
            for policy in ("fresh", "reuse") for i in range(len(s["turns"]))}
    return ids


def check_machine(payload):
    start = payload["state_start"]
    require(start.get("power_config") and start.get("supply"), "missing power state")
    initial_swap = number(start.get("swap_used_bytes"), "initial swap")
    for label in ("state_start", "state_last", "state_end"):
        state = payload[label]
        require(number(state.get("swap_used_bytes"), label + " swap") <= initial_swap,
                "swap grew during arm")
        for metric in ("process_rss_bytes", "process_peak_rss_bytes"):
            require(number(state.get(metric), label + " " + metric, True) <= 8 * 2**30,
                    "process RSS exceeds 8 GiB")
        require((state.get("power_config"), state.get("supply")) ==
                (start["power_config"], start["supply"]), "power state changed")
    # Every retained per-request RSS observation must also satisfy the limit.
    for item in payload.get("cases", []):
        for run in [item["warmup"], *item["runs"]]:
            require(number(run["memory"].get("process_peak_rss_bytes"), "request RSS", True)
                    <= 8 * 2**30, "request RSS exceeds 8 GiB")


def check_generation(record, prompt_count=None, cap=128, forced=False, cache_finite=False):
    tokens = record["output_token_ids"]
    require(isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens),
            "invalid output token IDs")
    count = len(tokens)
    require(count <= cap, "output exceeds cap")
    require(record["response_token_count"] == count and record["generated_tokens"] == count,
            "response/API count disagrees with output IDs")
    eos = record["eos_position_zero_based"]
    require(eos is None or type(eos) is int and eos == count - 1, "invalid EOS position")
    require(record["visible_non_eos_count"] == count - (eos is not None), "visible token count mismatch")
    require(record["finish_reason"] in ("stop", "length"), "invalid finish reason")
    require((record["finish_reason"] == "stop") == (eos is not None), "EOS/finish mismatch")
    if record["finish_reason"] == "length":
        require(count == cap, "length finish before cap")
    if forced:
        require(count == cap and eos is None and record["finish_reason"] == "length", "forced output mismatch")
    sha(record["generated_text_sha256"], "generated text")
    require(hashlib.sha256(record["generated_text"].encode()).hexdigest() == record["generated_text_sha256"],
            "text digest mismatch")
    sha(record["cached_token_history_sha256"], "cached token history")
    cache = record["cache"]
    require(len(cache) == 16, "wrong cache layer count")
    offset = record["expected_cache_offset"]
    require(type(offset) is int and offset >= count, "invalid expected cache offset")
    if prompt_count is not None:
        require(offset == prompt_count + count, "cache offset disagrees with prompt/output")
    for layer in cache:
        require(layer["type"] == "KVCache" and layer["offset"] == offset, "cache type/offset mismatch")
        require(len(layer["shapes"]) == 2 and all(len(s) == 4 and s[:2] == [1, 8]
                and s[2] >= offset and s[3] == 64 for s in layer["shapes"]), "invalid cache shape")
        number(layer["bytes"], "cache bytes", True)
    if cache_finite:
        require(record.get("cache_finite") is True, "cache finiteness unverified")
    for metric in METRICS:
        number(record[metric], metric, True)
    require(record["time_to_first_token_seconds"] <= record["request_seconds"], "TTFT exceeds request")
    memory = record["memory"]
    require(math.isclose(number(memory["peak_bytes"], "request peak", True) / 1e9,
                         record["mlx_peak_memory_gb"], rel_tol=1e-12), "request peak unit mismatch")
    number(memory["process_peak_rss_bytes"], "request RSS", True)


def check_packing(payload, numeric=False):
    arm = payload["arm"]
    if arm == "C":
        require(not payload.get("packing"), "control contains packing")
        return
    entries = payload["packing"]
    expected = {(i, family) for i in range(16) for family in ("Q", "G") if family in arm}
    keys = [(e["layer"], e["family"]) for e in entries]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "incomplete/duplicate packed families")
    for entry in entries:
        dims = [2048, 512, 512] if entry["family"] == "Q" else [8192, 8192]
        require(entry["output_dims"] == dims and entry["input_dims"] == 2048, "packed dimensions differ")
        require((entry["group_size"], entry["bits"], entry["mode"], entry["transpose"]) ==
                (64, 4, "affine", True), "packing quantization differs")
        require(entry["source_parameter_bytes"] == entry["packed_parameter_bytes"] > 0,
                "packed byte accounting differs")
        require(entry["original_projection_arrays_retained_in_model"] is False and
                entry["tied_head_preserved"] is True, "weight retention/tied-head evidence fails")
        require(entry["whole_model_parameter_bytes_before"] == entry["whole_model_parameter_bytes_after"]
                == payload["parameter_bytes_before"], "whole-model packing bytes differ")
        require(entry["split_indices"] == [sum(dims[:i]) for i in range(1, len(dims))], "split boundaries differ")
        components = entry["components"]
        require(set(components) == {"weight", "scales", "biases"}, "missing full packed component metadata")
        total_component_bytes = 0
        for name, component in components.items():
            require(component["shape"] == [sum(dims), 256 if name == "weight" else 32],
                    "full packed component shape differs")
            dtype = component["dtype"].split(".")[-1]
            sizes = {"uint32": 4} if name == "weight" else {"bfloat16": 2, "float16": 2, "float32": 4}
            require(dtype in sizes, "unsupported packed component dtype")
            require(component["nbytes"] == math.prod(component["shape"]) * sizes[dtype],
                    "packed component byte/shape accounting differs")
            total_component_bytes += component["nbytes"]
        require(total_component_bytes == entry["packed_parameter_bytes"], "packed component total bytes differ")
        require(components["scales"]["dtype"] == components["biases"]["dtype"], "affine parameter dtypes differ")
        parts = entry["partitions"]
        require(len(parts) == len(dims), "incomplete byte partitions")
        for i, part in enumerate(parts):
            require(part["partition"] == i and part["start"] == sum(dims[:i]) and
                    part["end"] == sum(dims[:i+1]), "packed row boundaries differ")
            require(set(part["components"]) == {"weight", "scales", "biases"}, "missing packed components")
            for name, component in part["components"].items():
                expected_shape = [dims[i], 256 if name == "weight" else 32]
                require(component["source"]["shape"] == expected_shape and
                        component["source"]["dtype"] == components[name]["dtype"] and
                        component["source"]["nbytes"] * sum(dims) == components[name]["nbytes"] * dims[i],
                        "source partition component accounting differs")
                require(component["bitwise_equal"] is True, "packed component bytes differ")
                for field in ("shape", "dtype", "nbytes"):
                    require(component["source"][field] == component["packed_slice"][field], "packed component metadata differs")
        if numeric:
            verification = entry["verification"]
            require(verification["passed"] is True, "numerical verification reports failure")
            rows = verification["checks"]
            expected_grid = {(n, kind, j) for n in ROWS for kind in
                             ("zeros", "ones", "random", "activation") for j in range(len(dims))}
            grid = [(v["rows"], v["input"], v["component"]) for v in rows]
            require(len(grid) == len(set(grid)) and set(grid) == expected_grid, "incomplete numerical matrix")
            for value in rows:
                require(value["layer"] == entry["layer"] and value["family"] == entry["family"],
                        "numerical evidence assigned to wrong family")
                require(value["finite"] is True and value["allclose"] is True, "numerical comparison fails")
                require(value["output_shape"] == [1, value["rows"], dims[value["component"]]], "numerical split shape differs")
                number(value["max_abs"], "numeric maximum")
                number(value["rms"], "numeric RMS")


def validate(payload, stage):
    require(payload["experiment"] == "experiment8-packed-projections" and payload["arm"] in ARMS,
            "wrong experiment/arm")
    require(payload["stage"] == stage, "wrong stage")
    expected_workload = "exp8-packed-" + {"screen": "screen-v1", "full": "diverse-v1", "correctness": "correctness-v1"}[stage]
    require(payload["workload_id"] == expected_workload, "wrong workload version")
    require(payload.get("completed_at_utc") and not payload.get("failure") and payload.get("status") == "pass",
            "capture incomplete, stopped, or rejected")
    require(stamp(payload["completed_at_utc"]) > stamp(payload["started_at_utc"]), "invalid run interval")
    require(payload["environment"]["mlx"] == "0.32.2" and payload["environment"]["mlx_lm"] == "0.31.3",
            "wrong pinned runtime")
    require(payload["environment"]["device"]["device_name"] == "Apple M2 Pro", "wrong device")
    settings = payload["settings"]
    require(settings["date"] == "04 Oct 2026" and settings["prefill_step_size"] == 2048 and
            settings["greedy"] is True and settings["fresh_primary_cache"] is True and
            settings["concurrency"] == 1, "measurement settings differ")
    manifest = payload["manifest"]
    required_sources = {"mlx_lm.models.llama", "mlx_lm.generate", "mlx.nn.layers.quantized",
        "experiments/experiment-8-packed-projections-plan.md", "experiments/experiment-8-packed-projections-spec.md",
        "benchmarks/run_experiment8_packed.py", "benchmarks/experiment8_packing.py", "benchmarks/run_baseline.py",
        "benchmarks/fixtures/experiment2-sessions.json"}
    require(required_sources <= set(manifest), "incomplete source/plan/spec manifest")
    for suffix in (".safetensors", "/config.json", "/tokenizer.json", "/tokenizer_config.json"):
        require(any(key.startswith("models/") and key.endswith(suffix) for key in manifest),
                "incomplete model/tokenizer manifest")
    require(list(settings["numerical_shapes"]) == list(ROWS) and settings["numerical_seed"] == 20261020,
            "numerical settings differ")
    for key, value in payload["manifest"].items():
        sha(value, key)
    require(payload["parameter_bytes_before"] == payload["parameter_bytes_after"] > 0, "retained parameter bytes changed")
    check_machine(payload)
    check_packing(payload, numeric=stage == "correctness")
    if stage == "correctness":
        require(set(payload["checks"]) == expected_correctness_ids(), "incomplete Stage-0 fixtures/histories")
        for key, check in payload["checks"].items():
            sha(check["prompt_sha256"], "prompt")
            if "prompt_token_ids" in check:
                require(digest(check["prompt_token_ids"]) == check["prompt_sha256"], "history prompt digest mismatch")
            check_generation(check["generation"], cap=256 if key == "forced-2048-256" else 128,
                             forced=key.startswith("forced-"), cache_finite=True)
            require(check["quality"] == evaluate_quality_check(key, check["generation"]["generated_text"]),
                    f"stored quality evidence differs from frozen validator: {key}")
        require(type(payload.get("absolute_quality_pass")) is bool, "absolute quality evidence missing")
        return
    cases = payload["cases"]
    keys = [c["id"] for c in cases]
    require(len(keys) == len(set(keys)) and set(keys) == expected_case_ids(stage), "incomplete/duplicate timing fixtures")
    repetitions = 5 if stage == "screen" else 10
    for case in cases:
        target = case["target_prompt_tokens"]
        require(target in TARGETS and f"-{target}-" in case["id"], "fixture target mismatch")
        require(type(case["actual_prompt_tokens"]) is int and case["actual_prompt_tokens"] > 0, "invalid prompt count")
        sha(case["prompt_sha256"], "prompt")
        require(len(case["runs"]) == repetitions, "incomplete/extra measured repetitions")
        reference = case["warmup"]
        for record in [reference, *case["runs"]]:
            check_generation(record, case["actual_prompt_tokens"], forced=stage == "screen")
            require(all(record[field] == reference[field] for field in IDENTITY), "nondeterministic warmup/measured outputs")
            require(record["cached_token_history_sha256"] == reference["cached_token_history_sha256"],
                    "nondeterministic cache token history")


def checked(payload, stage):
    try:
        validate(payload, stage)
        return []
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        return [str(error)]


def match_contract(control, candidate):
    for key in ("environment", "manifest", "settings", "workload_id"):
        require(control[key] == candidate[key], f"matched control {key} differs")
    require((control["state_start"]["power_config"], control["state_start"]["supply"]) ==
            (candidate["state_start"]["power_config"], candidate["state_start"]["supply"]), "matched arm power differs")


def quality_pass(payload):
    scope = payload.get("absolute_quality_scope", "")
    scope = scope.casefold() if isinstance(scope, str) else ""
    incomplete_scope = not scope or any(term in scope for term in
        ("unmeasured", "no extraction schema", "no semantic evaluation", "or semantic evaluation is claimed"))
    values = [value for check in payload.get("checks", {}).values() for value in check.get("quality", {}).values()]
    return (payload.get("absolute_quality_pass") is True and not incomplete_scope and bool(values)
            and all(value is not False and not (isinstance(value, str) and value.startswith("unmeasured"))
                    for value in values))


def prerequisite(arm, correctness):
    errors = []
    correctness = correctness or {}
    for name in ("C", arm):
        if name not in correctness:
            errors.append(f"missing completed Stage-0 evidence for {name}")
        else:
            errors += [f"Stage-0 {name}: {e}" for e in checked(correctness[name], "correctness")]
            if not quality_pass(correctness[name]):
                errors.append(f"Stage-0 {name}: absolute task quality failed or unmeasured")
    if errors:
        return errors
    try:
        control, candidate = correctness["C"], correctness[arm]
        match_contract(control, candidate)
        for key, check in control["checks"].items():
            other = candidate["checks"][key]
            require(check["prompt_sha256"] == other["prompt_sha256"], f"Stage-0 prompt differs: {key}")
            require(check["generation"]["cached_token_history_sha256"] == other["generation"]["cached_token_history_sha256"],
                    f"Stage-0 cached token history differs: {key}")
            require(all(check["generation"][f] == other["generation"][f] for f in IDENTITY),
                    f"Stage-0 output/cache differs: {key}")
    except (EvidenceError, KeyError, TypeError) as error:
        errors.append(str(error))
    return errors


def measurements(control, candidate):
    match_contract(control, candidate)
    require([c["id"] for c in control["cases"]] == [c["id"] for c in candidate["cases"]],
            "matched fixture order differs")
    cc = {c["id"]: c for c in control["cases"]}
    rc = {c["id"]: c for c in candidate["cases"]}
    per_fixture, by_length = {}, {}
    for key, case in cc.items():
        other = rc[key]
        require(case["prompt_sha256"] == other["prompt_sha256"] and
                case["actual_prompt_tokens"] == other["actual_prompt_tokens"], f"prompt differs: {key}")
        for record in [other["warmup"], *other["runs"]]:
            require(all(record[f] == case["warmup"][f] for f in IDENTITY), f"output/cache differs: {key}")
            require(record["cached_token_history_sha256"] == case["warmup"]["cached_token_history_sha256"],
                    f"cached token history differs: {key}")
        per_fixture[key] = {m: percent(statistics.median(r[m] for r in other["runs"]),
                                     statistics.median(r[m] for r in case["runs"])) for m in METRICS}
    for length in TARGETS:
        a = [r for c in cc.values() if c["target_prompt_tokens"] == length for r in c["runs"]]
        b = [r for c in rc.values() if c["target_prompt_tokens"] == length for r in c["runs"]]
        by_length[str(length)] = {"observations_per_arm": len(a),
            "median_change_percent": {m: percent(statistics.median(r[m] for r in b), statistics.median(r[m] for r in a)) for m in METRICS},
            "nearest_rank_p95_change_percent": {m: percent(nearest_rank([r[m] for r in b]), nearest_rank([r[m] for r in a])) for m in METRICS[:2]}}
    packing_extra = max(number(candidate["memory_loaded"]["peak_bytes"], "candidate load peak", True),
                        number(candidate["memory_packing"]["peak_bytes"], "candidate packing peak", True)) - number(control["memory_loaded"]["peak_bytes"], "control load peak", True)
    return per_fixture, by_length, packing_extra


def compare(control, candidate, stage, correctness=None):
    arm = candidate.get("arm")
    result = {"arm": arm, "stage": stage, "eligible": False, "errors": []}
    result["errors"] += checked(control, stage) + checked(candidate, stage)
    if control.get("arm") != "C" or arm not in ("Q", "G", "QG"):
        result["errors"].append("comparison requires C and a candidate arm")
    if result["errors"]:
        return result
    try:
        for payload in (control, candidate):
            source = (correctness or {}).get(payload["arm"])
            if source is not None:
                for key in ("manifest", "environment", "settings", "parameter_bytes_before", "parameter_bytes_after"):
                    require(source[key] == payload[key], f"timing and Stage-0 {key} differ for {payload['arm']}")
        fixtures, lengths, packing_extra = measurements(control, candidate)
    except (EvidenceError, KeyError, TypeError, ZeroDivisionError) as error:
        result["errors"].append(str(error))
        return result
    result.update(per_fixture_median_change_percent=fixtures, by_length=lengths,
                  extra_startup_peak_bytes=packing_extra)
    medians = {int(k): v["median_change_percent"] for k,v in lengths.items()}
    gates = {"matched_raw_output_cache_and_prompts": True,
             "request_memory_at_most_105_percent": all(allowed(v[METRICS[2]], 5) for v in medians.values()),
             "extra_startup_peak_at_most_512_mib": packing_extra <= 512 * 2**20}
    if stage == "screen":
        gates.update(short_request_improvement_at_least_5_percent=any(allowed(medians[n][METRICS[0]], -5) for n in (64,512)),
                     all_lengths_request_and_ttft_regression_at_most_5_percent=all(allowed(v[m], 5) for v in medians.values() for m in METRICS[:2]))
    else:
        gates.update(both_short_request_improvement_at_least_10_percent=all(allowed(medians[n][METRICS[0]], -10) for n in (64,512)),
                     per_fixture_request_and_ttft_regression_at_most_5_percent=all(allowed(v[m], 5) for v in fixtures.values() for m in METRICS[:2]),
                     long_request_and_ttft_regression_at_most_5_percent=all(allowed(medians[n][m], 5) for n in (2048,4096) for m in METRICS[:2]),
                     all_lengths_p95_request_and_ttft_regression_at_most_10_percent=all(allowed(v, 10) for group in lengths.values() for v in group["nearest_rank_p95_change_percent"].values()))
    result["prerequisite_errors"] = prerequisite(arm, correctness)
    # Explicit false on timing payloads is still a blocker, even with Stage-0 supplied.
    if control.get("absolute_quality_pass") is False or candidate.get("absolute_quality_pass") is False:
        result["prerequisite_errors"].append("timing payload explicitly reports absolute quality failure")
    gates["stage0_correctness_and_absolute_quality"] = not result["prerequisite_errors"]
    result["gates"] = gates
    result["mechanism_screen_pass"] = all(v for k,v in gates.items() if k != "stage0_correctness_and_absolute_quality")
    result["eligible"] = all(gates.values())
    result["selection_key"] = [statistics.mean(medians[n][METRICS[0]] for n in (64,512)),
                               max(medians[n][m] for n in (2048,4096) for m in METRICS[:2]),
                               len(arm), ARMS.index(arm)]
    return result


def analyze_screen(payloads, correctness=None):
    arms = [p.get("arm") for p in payloads]
    require(len(arms) == len(set(arms)), "duplicate screen arm")
    require(arms.count("C") == 1, "screen requires one control")
    control = payloads[arms.index("C")]
    candidates = [compare(control, p, "screen", correctness) for p in payloads if p.get("arm") != "C"]
    finalists = sorted((c for c in candidates if c["eligible"]), key=lambda c: c["selection_key"])
    missing = [arm for arm in ARMS if arm not in arms]
    # Missing registered candidate arms require explicit Stage-0 rejection evidence.
    unresolved = []
    for arm in missing:
        evidence = (correctness or {}).get(arm, {})
        if evidence.get("status") != "rejected" or not evidence.get("failure"):
            unresolved.append(arm)
    return {"stage": "screen", "candidates": candidates, "missing_arms": missing,
            "unresolved_arms": unresolved,
            "finalist": finalists[0]["arm"] if finalists and not unresolved else None,
            "selection_frozen": not unresolved,
            "selection_rule": "mean short request change; worst long request/TTFT regression; family count; Q/G/QG rank"}


def analyze_pair(control, candidate, correctness=None):
    return compare(control, candidate, "full", correctness)


def analyze_confirmation(pairs, selected_arm, correctness=None):
    require(selected_arm in ("Q", "G", "QG"), "a frozen screen finalist is required")
    results = [analyze_pair(c, f, correctness) for c,f in pairs]
    errors = []
    if len(pairs) != 2:
        errors.append("exactly two independent full pairs required")
    elif any(f.get("arm") != selected_arm or c.get("arm") != "C" for c,f in pairs):
        errors.append("full pair arm differs from frozen finalist; no fallback finalist allowed")
    else:
        ordered = [pairs[0][0], pairs[0][1], pairs[1][1], pairs[1][0]]
        try:
            for a,b in zip(ordered, ordered[1:]):
                require(stamp(a["completed_at_utc"]) <= stamp(b["started_at_utc"]),
                        "full repeats overlap, reuse a capture, or violate C/finalist then finalist/C order")
            for p in ordered[1:]:
                match_contract(ordered[0], p)
        except (EvidenceError, KeyError, TypeError) as error:
            errors.append(str(error))
    passed = [r["eligible"] for r in results]
    quality_limited = any(any("quality" in e for e in r.get("prerequisite_errors", [])) for r in results)
    outcome = "inconclusive" if errors or quality_limited or any(r["errors"] for r in results) or len(set(passed)) > 1 else ("select" if passed and all(passed) else "reject")
    return {"outcome": outcome, "selected_arm": selected_arm if outcome == "select" else None,
            "frozen_finalist": selected_arm, "pairs": results, "errors": errors,
            "repeat_results_combined": False,
            "absolute_quality_limitation": quality_limited,
            "validator_limitations": ["Whole-word Monday can reject the valid plural Mondays.",
                "The strict three-line bullet validator can reject an introduction plus three valid bullets; this does not establish an item-count failure.",
                "The extraction validator checks nonempty complete JSON, without a semantic schema."]}


def read_payload(path):
    """Read an original JSON or losslessly archived .json.gz result offline."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as source:
            return json.load(source)
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, nargs="+")
    parser.add_argument("--pair", type=Path, nargs=2, action="append", default=[])
    parser.add_argument("--correctness", type=Path, nargs="+", default=[])
    parser.add_argument("--finalist", choices=("Q", "G", "QG"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load = read_payload
    correctness = {}
    for path in args.correctness:
        payload = load(path)
        require(payload["arm"] not in correctness, "duplicate correctness arm")
        correctness[payload["arm"]] = payload
    result = {"experiment": "experiment8-packed-projections-offline-analysis",
              "correctness_validation": {arm: {"errors": checked(p,"correctness"),
                  "absolute_quality_pass": quality_pass(p)} for arm,p in correctness.items()}}
    finalist = args.finalist
    if args.screen:
        result["screen"] = analyze_screen([load(p) for p in args.screen], correctness)
        if finalist is not None:
            require(finalist == result["screen"]["finalist"], "--finalist differs from screen selection")
        finalist = result["screen"]["finalist"]
    if args.pair:
        require(finalist is not None, "--pair requires frozen screen selection or --finalist")
        result["confirmation"] = analyze_confirmation([(load(c),load(f)) for c,f in args.pair], finalist, correctness)
    require(args.screen or args.pair or args.correctness, "no evidence supplied")
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as out:
            out.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

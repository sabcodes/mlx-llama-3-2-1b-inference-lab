#!/usr/bin/env python3
"""Isolate M2 prefill, decode, or KV updates for a bounded Instruments capture."""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

from run_baseline import ROOT, make_fixture, write_json
from run_experiment7_profile import PROFILE_CHAT_TEMPLATE_DATE, consume_direct_generation


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "decode", "kv"), required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    for path in (args.ready, args.progress, args.output):
        if not path.resolve().is_relative_to(ROOT) or path.exists():
            parser.error("Use new paths inside this project")

    from mlx_lm import load
    from mlx_lm.generate import generate_step, generation_stream
    from mlx_lm.models.cache import make_prompt_cache, KVCache
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    import mlx.core as mx

    def synchronize():
        # generate_step submits asynchronous work on its dedicated stream.
        mx.synchronize(generation_stream)
        mx.synchronize()

    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    lib.mach_absolute_time.restype = ctypes.c_uint64
    ticks = lib.mach_absolute_time
    triggered = released = False

    def trigger(signum, frame):
        nonlocal triggered
        triggered = True

    def release(signum, frame):
        nonlocal released
        released = True

    signal.signal(signal.SIGUSR1, trigger)
    signal.signal(signal.SIGUSR2, release)

    def wait_for(which):
        deadline = time.monotonic() + 180
        while not args.validate_only and not (triggered if which == "trigger" else released):
            if time.monotonic() > deadline:
                raise TimeoutError(which + " timeout")
            time.sleep(.02)

    model, tokenizer = load(str(ROOT / "models/Llama-3.2-1B-Instruct-4bit"))
    fixture = make_fixture(tokenizer, 2048, "summarization", PROFILE_CHAT_TEMPLATE_DATE)
    prompt = fixture["token_ids"]
    sampler = make_sampler(temp=0.0)
    processors = make_logits_processors(logit_bias={t: -1e9 for t in tokenizer.eos_token_ids})
    common = dict(mx=mx, generate_step=generate_step, make_prompt_cache=make_prompt_cache,
                  sampler=sampler, logits_processors=processors, token_ids=prompt, prefill_step_size=2048)
    consume_direct_generation(model=model, max_tokens=8, **common)
    synchronize()
    result = {"schema_version": 1, "experiment": "experiment-7-m2-phase-diagnostic",
              "phase": args.phase, "classification": "instrumented diagnostic; not clean latency",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "actual_prompt_tokens": len(prompt), "chat_template_date": PROFILE_CHAT_TEMPLATE_DATE,
              "prompt_token_ids_sha256": digest(prompt), "windows": [],
              "mlx_peak_memory_bytes": 0}

    def ready():
        synchronize()
        write_json(args.ready, {"pid": __import__("os").getpid(), "warmup_completed": True})
        wait_for("trigger")

    def done():
        synchronize()
        write_json(args.progress, {"recorded_work_completed": True, "windows": result["windows"]})
        wait_for("release")

    if args.phase == "kv":
        heads, dim, layers = model.args.num_key_value_heads, model.args.head_dim, len(model.layers)
        specs = [("initial_allocation", 0, 0, len(prompt)-1),
                 ("ordinary_append", len(prompt)+4, 2304, 1),
                 ("growth_at_2048", 2048, 2048, 1)]
        prepared = []
        for name, offset, capacity, count in specs:
            keys = mx.ones((1, heads, count, dim), dtype=mx.float16)
            values = mx.full(keys.shape, 2, dtype=mx.float16)
            trials = []
            for _ in range(5):
                caches = [KVCache() for _ in range(layers)]
                for cache in caches:
                    if capacity:
                        cache.keys = mx.zeros((1, heads, capacity, dim), dtype=mx.float16)
                        cache.values = mx.zeros_like(cache.keys)
                        cache.offset = offset
                trials.append(caches)
            mx.eval(keys, values, [[c.state for c in cs] for cs in trials] if capacity else [])
            prepared.append((name, offset, capacity, count, keys, values, trials))
        # Compile the three exact KV paths outside the recording.
        for name, offset, capacity, count, keys, values, trials in prepared:
            cache = KVCache()
            if capacity:
                cache.keys = mx.zeros((1, heads, capacity, dim), dtype=mx.float16)
                cache.values = mx.zeros_like(cache.keys)
                cache.offset = offset
            cache.update_and_fetch(keys, values)
            mx.eval(cache.state)
        synchronize()
        ready()
        for name, offset, capacity, count, keys, values, trials in prepared:
            window = {"name": name, "start_mach_absolute": ticks(), "trials": [],
                      "layers_per_trial": layers, "offset_before": offset,
                      "capacity_before": capacity, "input_shape": list(keys.shape),
                      "input_dtype": "float16", "synthetic_inputs_evaluated_before_recording": True}
            for caches in trials:
                began = time.perf_counter()
                for cache in caches:
                    cache.update_and_fetch(keys, values)
                mx.eval([c.state for c in caches])
                synchronize()
                window["trials"].append({"synchronized_update_seconds": time.perf_counter()-began})
            window["end_mach_absolute"] = ticks()
            result["windows"].append(window)
        done()
        # Numerical validation is outside the recorded windows.
        for name, offset, capacity, count, keys, values, trials in prepared:
            for caches in trials:
                for cache in caches:
                    assert cache.offset == offset+count
                    assert bool(mx.all(cache.keys[..., offset:offset+count, :] == 1))
                    assert bool(mx.all(cache.values[..., offset:offset+count, :] == 2))
                    if offset:
                        assert bool(mx.all(cache.keys[..., :offset, :] == 0))
                        assert bool(mx.all(cache.values[..., :offset, :] == 0))
        result["kv_numerical_checks_passed"] = True
        result["kv_source_path"] = "mlx_lm/models/cache.py:KVCache.update_and_fetch"
        result["limitations"] = ["Evaluated synthetic inputs use actual model/cache geometry; projection, RoPE and attention are excluded.",
                                  "Synchronized isolated costs do not estimate a fraction of ordinary request latency."]
    else:
        cache = make_prompt_cache(model)
        calls = []

        class BoundaryModel:
            def __call__(self, inputs, cache):
                if args.phase == "prefill":
                    if not calls:
                        ready()
                        result["windows"].append({"name": "prefill_and_first_token", "start_mach_absolute": ticks()})
                    elif len(calls) == 2:
                        synchronize()
                        result["windows"][0]["end_mach_absolute"] = ticks()
                        result["cache_offsets_at_phase_end"] = [c.offset for c in cache]
                        done()
                calls.append(inputs.shape[1])
                return model(inputs, cache=cache)

        generator = generate_step(mx.array(prompt), BoundaryModel(), max_tokens=128,
                                  prompt_cache=cache, prefill_step_size=2048,
                                  sampler=sampler, logits_processors=processors)
        tokens = []
        if args.phase == "decode":
            tokens = [int(next(generator)[0]) for _ in range(4)]
            synchronize()
            ready()
            offsets = [c.offset for c in cache]
            start_call = len(calls)
            window = {"name": "decode_32_model_steps", "start_mach_absolute": ticks(),
                      "cache_offsets_at_start": offsets}
            tokens += [int(next(generator)[0]) for _ in range(32)]
            synchronize()
            window["end_mach_absolute"] = ticks()
            window["model_input_lengths"] = calls[start_call:]
            window["cache_offsets_at_end"] = [c.offset for c in cache]
            assert window["model_input_lengths"] == [1]*32
            result["windows"].append(window)
            done()
        tokens += [int(t) for t, _ in generator]
        synchronize()
        expected = json.loads((ROOT / "benchmarks/results/experiment7-m2pro-phase-confirmation-20261019-r2.json").read_text())
        reference = next(c for c in expected["cases"] if c["actual_prompt_tokens"] == len(prompt))
        result["generated_tokens"] = len(tokens)
        result["output_token_ids_sha256"] = digest(tokens)
        result["output_matches_clean_m2"] = digest(tokens) == reference["runs"][0]["output_token_ids_sha256"]
        assert len(tokens) == 128 and result["output_matches_clean_m2"]
        assert calls[:2] == [len(prompt)-1, 1]
        result["limitations"] = ["The unchanged generator is paused only at synchronized capture boundaries; no clean-latency comparison is made.",
                                  "Prefill includes last-prompt-token and first-output sampling; no decode model call crosses its end boundary.",
                                  "Decode primes four yielded outputs and one evaluated lookahead outside capture; 32 new single-token model calls are inside."]
    result["mlx_peak_memory_bytes"] = mx.get_peak_memory()
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in result if k not in ("windows", "limitations")}), flush=True)


if __name__ == "__main__":
    main()

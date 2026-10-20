#!/usr/bin/env python3
"""Warm the existing experiment 7 workload, then wait for a CLI trace trigger."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from run_baseline import ROOT, make_fixture, write_json
from run_experiment7_profile import PROFILE_CHAT_TEMPLATE_DATE, consume_direct_generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-for-release", action="store_true")
    args = parser.parse_args()
    for path in (args.ready, args.output):
        if not path.resolve().is_relative_to(ROOT) or path.exists():
            parser.error("Use new output paths inside the project")

    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    import mlx.core as mx

    triggered = False
    released = False

    def start(signum, frame):
        nonlocal triggered
        triggered = True

    def release(signum, frame):
        nonlocal released
        released = True

    signal.signal(signal.SIGUSR1, start)
    signal.signal(signal.SIGUSR2, release)
    model, tokenizer = load(str(ROOT / "models/Llama-3.2-1B-Instruct-4bit"))
    fixture = make_fixture(tokenizer, 2048, "summarization", PROFILE_CHAT_TEMPLATE_DATE)
    common = dict(model=model, mx=mx, generate_step=generate_step,
                  make_prompt_cache=make_prompt_cache, sampler=make_sampler(temp=0.0),
                  logits_processors=make_logits_processors(
                      logit_bias={token: -1e9 for token in tokenizer.eos_token_ids}),
                  token_ids=fixture["token_ids"], prefill_step_size=2048)
    consume_direct_generation(max_tokens=8, **common)
    write_json(args.ready, {"pid": os.getpid(), "actual_prompt_tokens": fixture["actual_prompt_tokens"],
                           "warmup_completed": True, "chat_template_date": PROFILE_CHAT_TEMPLATE_DATE})
    deadline = time.monotonic() + 120
    while not triggered:
        if time.monotonic() > deadline:
            raise TimeoutError("No trace-start trigger within 120 seconds")
        time.sleep(0.05)

    result = {"classification": "instrumented diagnostic; not clean benchmark timing",
              "actual_prompt_tokens": fixture["actual_prompt_tokens"], "runs": []}
    for _ in range(3):
        began = time.perf_counter()
        tokens = consume_direct_generation(max_tokens=128, **common)
        result["runs"].append({"seconds": time.perf_counter() - began,
                              "generated_tokens": len(tokens),
                              "output_token_ids_sha256": hashlib.sha256(
                                  json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()})
        write_json(args.output, result)

    if args.hold_for_release:
        deadline = time.monotonic() + 180
        while not released and time.monotonic() < deadline:
            time.sleep(0.05)


if __name__ == "__main__":
    main()

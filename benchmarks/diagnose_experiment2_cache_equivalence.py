#!/usr/bin/env python3
"""Diagnose exact-output divergence between fresh and reused MLX-LM KV caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_experiment2 import (
    MODEL_ID,
    MODEL_PATH,
    PREFILL_STEP_SIZE,
    build_fixtures,
    render_conversation,
)


def find_session(document: dict, fixture_id: str) -> dict:
    return next(
        session for session in document["sessions"]
        if session["fixture_id"] == fixture_id and session["warmup"]
    )


def first_difference(left: list[int], right: list[int]) -> int:
    return next(
        (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)),
    )


def generate_with_trace(
    *, model, tokenizer, mx, stream_generate, sampler, prompt_tokens,
    prompt_cache, candidate_ids: tuple[int, int], max_tokens: int,
) -> tuple[list[int], list[dict]]:
    output_ids = []
    trace = []
    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
        prefill_step_size=PREFILL_STEP_SIZE,
    ):
        output_ids.append(int(response.token))
        trace.append({
            "token": int(response.token),
            "candidate_logprobs": {
                str(token): float(response.logprobs[token].item()) for token in candidate_ids
            },
        })
    return output_ids, trace


def build_prior_turn_cache(
    *, model, mx, make_prompt_cache, prior_prompt: list[int], prior_output: list[int]
):
    cache = make_prompt_cache(model)
    processed = 0
    while len(prior_prompt) - processed > 1:
        count = min(PREFILL_STEP_SIZE, len(prior_prompt) - processed - 1)
        segment = prior_prompt[processed : processed + count]
        model(mx.array(segment)[None], cache=cache)
        mx.eval([item.state for item in cache])
        mx.clear_cache()
        processed += count

    # The last prior-prompt token and generated tokens were evaluated one at a
    # time by generate_step. Replaying the same token IDs with the same shapes
    # reconstructs that cache without relying on an LRU lookup.
    for token in prior_prompt[processed:] + prior_output:
        model(mx.array([token])[None], cache=cache)
        mx.eval([item.state for item in cache])
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-json", type=Path, required=True)
    parser.add_argument("--reuse-json", type=Path, required=True)
    parser.add_argument("--fixture-id", default="repair-workshop")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    control_document = json.loads(args.control_json.read_text())
    reuse_document = json.loads(args.reuse_json.read_text())
    control_session = find_session(control_document, args.fixture_id)
    reuse_session = find_session(reuse_document, args.fixture_id)
    if len(reuse_session["turns"]) < 2:
        raise ValueError("Reuse artifact must include the divergent second turn")

    saved_control = control_session["turns"][1]["generated_token_ids"]
    saved_reuse = reuse_session["turns"][1]["generated_token_ids"]
    divergence = first_difference(saved_control, saved_reuse)
    if divergence >= min(len(saved_control), len(saved_reuse)):
        raise ValueError("Saved artifacts do not contain a token divergence")
    candidate_ids = (saved_control[divergence], saved_reuse[divergence])

    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    model, tokenizer = load(str(MODEL_PATH))
    _, fixtures = build_fixtures(tokenizer)
    fixture = next(item for item in fixtures if item["id"] == args.fixture_id)
    first_user = {"role": "user", "content": fixture["turns"][0]["user"]}
    first_assistant = {
        "role": "assistant",
        "content": control_session["turns"][0]["output_text"],
    }
    second_user = {"role": "user", "content": fixture["turns"][1]["user"]}
    prior_prompt = render_conversation(tokenizer, fixture["system_prompt"], [first_user])
    full_second_prompt = render_conversation(
        tokenizer, fixture["system_prompt"], [first_user, first_assistant, second_user]
    )
    prior_output = control_session["turns"][0]["generated_token_ids"]
    prefix_tokens = prior_prompt + prior_output
    if full_second_prompt[: len(prefix_tokens)] != prefix_tokens:
        raise AssertionError("Re-rendered second turn does not begin with the prior cache key")
    suffix = full_second_prompt[len(prefix_tokens) :]

    sampler = make_sampler(temp=0.0)
    fresh_ids, fresh_trace = generate_with_trace(
        model=model,
        tokenizer=tokenizer,
        mx=mx,
        stream_generate=stream_generate,
        sampler=sampler,
        prompt_tokens=full_second_prompt,
        prompt_cache=make_prompt_cache(model),
        candidate_ids=candidate_ids,
        max_tokens=control_document["settings"]["max_new_tokens"],
    )
    segmented_cache = build_prior_turn_cache(
        model=model,
        mx=mx,
        make_prompt_cache=make_prompt_cache,
        prior_prompt=prior_prompt,
        prior_output=prior_output,
    )
    segmented_ids, segmented_trace = generate_with_trace(
        model=model,
        tokenizer=tokenizer,
        mx=mx,
        stream_generate=stream_generate,
        sampler=sampler,
        prompt_tokens=suffix,
        prompt_cache=segmented_cache,
        candidate_ids=candidate_ids,
        max_tokens=control_document["settings"]["max_new_tokens"],
    )

    result = {
        "fixture_id": args.fixture_id,
        "model_id": MODEL_ID,
        "prefill_step_size": PREFILL_STEP_SIZE,
        "prior_prompt_tokens": len(prior_prompt),
        "prior_generated_tokens": len(prior_output),
        "reused_prefix_tokens": len(prefix_tokens),
        "second_turn_suffix_tokens": len(suffix),
        "saved_divergence_index": divergence,
        "candidate_token_ids": list(candidate_ids),
        "fresh_matches_saved_control": fresh_ids == saved_control,
        "segmented_matches_saved_reuse": segmented_ids == saved_reuse,
        "fresh_matches_segmented": fresh_ids == segmented_ids,
        "fresh_output_sha256": hashlib.sha256(json.dumps(fresh_ids).encode()).hexdigest(),
        "segmented_output_sha256": hashlib.sha256(json.dumps(segmented_ids).encode()).hexdigest(),
        "fresh_at_divergence": fresh_trace[divergence],
        "segmented_at_divergence": segmented_trace[divergence],
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

# Experiment 3: compiled-decode feasibility result

Date: 2026-10-03

## Decision

Reject the tested explicit `mx.compile` decode seam for this project's pinned MLX 0.32.2 / MLX-LM 0.31.3 runtime. Keep MLX-LM's current eager generation path.

This is a useful negative result, not a claim that `mx.compile` is generally ineffective. The problem is the interaction between this compilation boundary and MLX-LM's dynamically growing KV-cache state.

## Stage A design

The bounded probe used the local Llama 3.2 1B 4-bit model, a 515-token rendered prompt, eager prefill with step size 2,048, greedy sampling, and at most 32 decoded tokens. It compared the current eager single-token model/log-softmax/argmax path with an explicit compiled version that passed KV states in and returned updated states.

The probe was intentionally bounded. Its stop conditions included compilation failure, cache corruption, output mismatch, repeated tracing as token position changed, or unbounded/persistent memory growth.

## Results

### Shapeless compilation

Artifact: `experiment3-compile-probe-20261003T-stage-a-retry1.json`

The supported `shapeless=True` mode failed before generation with:

```text
[Primitive:output_shapes] Slice cannot infer output shapes.
```

The KV cache exposes progressively sliced key/value tensors. This pinned runtime could not infer those dynamic slice output shapes for the compiled graph.

### Shape-specific compilation

Artifact: `experiment3-compile-probe-20261003T-shaped-8.json`

The fallback was capped at eight tokens to prevent a bad trace strategy from accumulating unchecked.

| Metric | Eager | Compiled | Result |
| --- | ---: | ---: | --- |
| Generated tokens | 8 | 8 | exact token match |
| Final cache length | 522 | 522 | correct |
| Python trace calls | not applicable | 8 | one trace per token |
| First token | 0.0482 s | 0.8184 s | compiled about 17× slower |
| Subsequent decode | 150.2 tokens/s | 94.8 tokens/s | compiled 36.9% slower |
| MLX peak memory | 721,892,634 bytes | 798,704,828 bytes | compiled 10.6% higher |

The exact tokens and final cache length prove that the explicit-state implementation was functionally correct for this bounded case. The trace counter proves that each new cache shape caused a new compilation. This structural behavior is more important than small timing noise in a single diagnostic run.

## Why the full suite was stopped

The Stage A plan required stopping if token position caused repeated compilation or persistent trace growth. Shape-specific compilation traced eight times for eight tokens, already missed the memory budget, and was materially slower. Extending that behavior to 128-token responses and hundreds of requests would not answer a useful uncertainty and could retain many compiled graphs.

Therefore Stage B, the full 20-fixture suite, and the cumulative KV-cache chat confirmation were not run.

## What would justify revisiting this

Retest only if one of these changes creates a genuinely different compilation seam:

- MLX-LM adds an officially supported compiled decoder with correct cache-state handling;
- MLX gains a dynamic-cache representation that avoids per-position traces;
- a fixed-shape/preallocated cache can carry its offset as compiled array state without Python scalar specialization; or
- profiling identifies a smaller stateless decode subgraph whose compilation can produce a meaningful end-to-end improvement.

An MLX or MLX-LM upgrade would be a new dependency experiment and should not overwrite this result.

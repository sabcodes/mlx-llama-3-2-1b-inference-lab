# Experiment 6 analysis: speculative decoding

Date: 2026-10-04

## Decision

Reject speculative decoding with the existing Mixed-3/6 full-1B draft for the
current Llama 3.2 1B interactive-chat configuration. The implementation passed
the exact-token correctness gate, but every speculative arm was substantially
slower than target-only generation and used much more resident memory.

No candidate qualified for the full or confirmation captures, and no deployed
setting was changed.

This is a rejection of this **target/draft pairing**, not of speculative
decoding in general. The draft retains the target's complete 16-layer, 1B
architecture and is only 14.5% smaller in stored weights. A genuinely smaller,
purpose-built tokenizer-compatible draft can have a different result.

## Controlled setup

- Target: Llama 3.2 1B Instruct affine-4/group-64.
- Draft: the Experiment 4 Mixed-3/6/group-64 artifact.
- Target and draft `tokenizer.json` files were byte-identical.
- Runtime: MLX 0.32.2 and MLX-LM 0.31.3 in the project `.venv`.
- Greedy generation, concurrency one, and prefill step size 2,048.
- Separate fresh processes for target-only, zero-draft, and each draft count.
- Stage 1: 64-, 512-, and 4,096-token inputs; natural short answers capped at
  32 tokens and forced 128-token answers; one warmup plus three measured runs.
- Clean timing only; no `powermetrics` instrumentation.

## Stage 0: exact-token correctness

Draft counts 1, 2, 3, and 4 each matched ordinary target-only generation on
all seven cases. The gate covered five baseline prompt types at approximately
512 input tokens plus two forced 256-token generations intended to expose late
divergence. Token IDs, generated length, finish reason, EOS position, and
cache consistency all passed.

This is stronger than semantic similarity for the tested cases. It does not
prove that MLX-LM 0.31.3 can never diverge on a different prompt or longer
generation.

## Stage 1: bounded performance screen

Changes below are pooled medians for the forced 128-token workload versus the
target-only process. Positive latency means slower; negative decode throughput
means fewer tokens per second.

| Arm | TTFT change | Request-time change | Decode tok/s change | Draft-origin fraction | Reconstructed acceptance |
|---|---:|---:|---:|---:|---:|
| Zero draft | +83.36% | +65.29% | -38.87% | 0% | 0% |
| Draft 1 | +97.45% | +112.04% | -58.67% | 43.36% | 73.00% |
| Draft 2 | +94.11% | +114.50% | -59.40% | 58.59% | 71.12% |
| Draft 3 | +123.58% | +159.77% | -66.85% | 64.84% | 59.90% |
| Draft 4 | +128.77% | +186.03% | -70.85% | 69.53% | 57.67% |

The short-answer result was also uniformly negative. Median request time at the
32-token cap regressed by 89.10% for zero draft and by 88.29% to 146.18% for
the four real draft arms.

All Stage-1 outputs remained exactly token-identical to the target-only output.
The failure is performance and memory, not answer quality.

## Memory cost

Target-only active MLX memory immediately after model load was 0.695 GB. With
both target and draft resident, it was 1.289 GB: an 85.5% increase before any
request cache or temporary work.

Per-request peak-memory growth depended on context length. It was about 76.3%
higher at 64 input tokens, 49.7% at 512, and 44.6% at 4,096. The percentage
falls at longer contexts because the target-only control's own prompt cache and
prefill temporaries grow; the draft still adds roughly the same model-weight
burden.

The 64-token cases therefore also failed the approved 50% peak-memory ceiling.

## Why it lost despite useful acceptance

The one- and two-token arms accepted roughly 70% of reconstructed proposals,
which would be encouraging for a cheap draft. This draft is not cheap: it runs
the same 16-layer, 1B transformer shape as the target with a different weight
encoding. MLX-LM also prefills the draft and target sequentially and maintains
separate caches.

The zero-draft control demonstrates the underlying burden. Merely loading both
models and using the speculative path, without proposing a token, already made
128-token requests 65.3% slower. Actual drafting added more work than parallel
verification could recover. Larger draft groups increased the emitted
draft-token share but reduced proposal acceptance and worsened latency.

## Gate result

| Arm | Exact tokens | Plausible 5% decode gain | Request regression <=10% | Memory guardrails | Outcome |
|---|---|---|---|---|---|
| Zero draft | Pass | Fail | Fail | Fail at 64 tokens | Reject/control only |
| Draft 1 | Pass | Fail | Fail | Fail at 64 tokens | Reject |
| Draft 2 | Pass | Fail | Fail | Fail at 64 tokens | Reject |
| Draft 3 | Pass | Fail | Fail | Fail at 64 tokens | Reject |
| Draft 4 | Pass | Fail | Fail | Fail at 64 tokens | Reject |

The screen used three measured repetitions and is intended for elimination,
not a high-precision estimate of the exact slowdown. The losses were large,
consistent across context lengths, and opposite the advancement threshold, so
a ten-repeat full suite could not establish the required 20% decode win.

## Raw evidence

- Stage-0 correctness: `experiment6-stage0-mixed3_6-20261004.json`.
- Target-only Stage 1: `experiment6-stage1-target-20261004.json`.
- Zero-draft control: `experiment6-stage1-zero-draft-20261004.json`.
- Draft-count screens: `experiment6-stage1-draft1-20261004.json`,
  `experiment6-stage1-draft2-20261004.json`,
  `experiment6-stage1-draft3-20261004.json`, and
  `experiment6-stage1-draft4-20261004.json`.

## What we learned

Speculative decoding needs more than accurate guesses. The draft must be much
cheaper than the target. Re-encoding the same 1B architecture at slightly lower
precision is not enough on this M2 Pro; the additional model traffic, dual
prefill, cache, and verification overhead dominate.

A future speculative-decoding experiment should begin only after obtaining or
building a substantially smaller draft that shares the target tokenizer, or a
purpose-built self-speculative implementation that reuses target layers and
cache rather than loading another full model.

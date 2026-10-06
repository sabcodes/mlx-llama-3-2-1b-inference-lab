# Experiment 5 post-review corrections

Date: 2026-10-05

## Purpose

This addendum preserves the independent scientific, specification, and
standards reviews performed after Experiment 5 was initially committed. The
main analysis has been corrected without rewriting the original raw JSON or
Git history.

## Corrected disposition

Post-prefill 8-bit KV-cache quantization is **rejected under the original
success rules**, not a passing research-only candidate. The overall deployment
decision remains unchanged: do not enable KV-cache quantization for the current
1B interactive-chat configuration.

Two independent hard gates failed:

1. Whole-request peak MLX memory at 16K decreased by 0%, below the required
   20% reduction.
2. At 8K, the unquantized control produced the required three-bullet grounded
   summary while post-prefill 8-bit produced two bullets. That is a new
   required-format failure, which the approved quality rule forbids.

The arm did pass the direct-cache-byte target and preserved the focused fact
retrieval checks. Those useful mechanism results do not override the failed
quality and peak-memory gates.

## Additional review findings

### Aggregate score naming

The historical result field `required_fact_score` divides all passed evaluator
checks by all checks. Its numerator includes fact, formatting, valid-JSON, and
forbidden-phrase checks. The numerical values are correct, but the field is
misleadingly named and must be interpreted as an aggregate-check score.

### Structured JSON evaluation

The structured-extraction fixture checked whether expected strings appeared in
serialized JSON. It did not assert the exact key/value mapping, so some
incorrect objects could pass. Future quality suites should compare the parsed
object with the expected structure and values.

### Missing planned workload

The approved plan called for a multi-turn-like history-recall case. The runner
rendered each implemented fixture as one system message plus one user message
containing synthetic notes; no prior user/assistant turns were tested.
Experiment 5 therefore has incomplete quality coverage for multi-turn chat.

### Exploratory speed evidence

The quality captures had no warmup or repetitions. At 8K, one 8-bit candidate
response generated 72 tokens while the corresponding control generated 96, so
that comparison was not identical work. The observed decode-throughput signal
is hypothesis-generating only and was correctly not promoted to a confirmed
speed result.

## What remains valid

- Cache-byte calculations, cache types, layer counts, and offsets recomputed
  correctly.
- Post-prefill 8-bit reduced logical cache bytes by 46.875% at 16K and active
  MLX memory by 20.536% while reducing whole-request peak memory by 0%.
- Moving conversion to the prompt boundary avoided the immediate-quantization
  prefill memory explosion.
- Stopping before repeated performance runs was justified because no candidate
  remained eligible for adoption.

## Future protocol changes

- Report fact, format, contradiction, and structural-JSON checks separately.
- Compare candidates against control per mandatory check before considering an
  aggregate score.
- Add an actual multi-turn history fixture when KV-cache quantization is
  revisited.
- Require repeated identical-output-length timing before claiming a decode
  improvement.

# Experiment 5: KV-cache quantization for long-context chat

## Plain-language question

Can MLX store the model's temporary memory of a long conversation in fewer
bits, substantially reducing memory use without losing supplied facts or making
interactive generation noticeably slower?

This experiment does not change or retrain the model weights. It changes only
the key/value (KV) cache created while processing and generating tokens.

## Scope confirmed by the user

- Hardware: the current M2 Pro MacBook Pro with 16 GB unified memory.
- Model: the existing Llama 3.2 1B Instruct affine-4/group-64 checkpoint.
- Runtime: the project-local `.venv`, MLX 0.32.2, MLX-LM 0.31.3.
- Use case: single-user interactive chat.
- Context lengths: 4,096, 8,192, and 16,384 tokens.
- Generation: greedy, 128-token cap, concurrency one.
- Selected prompt-prefill step size: 2,048.
- Clean timing runs remain separate from optional system telemetry.

## Arms

1. Unquantized BF16 KV cache (control).
2. 8-bit KV cache, group size 64, quantization start 0.
3. 4-bit KV cache, group size 64, quantization start 0.

Starting at token zero isolates the cache representation as the changed
variable and makes every tested length exercise quantization. If a candidate
saves meaningful retained memory but harms prefill, a separately labeled
second-stage candidate may keep prefill unquantized and convert the cache once
at the prompt boundary. The delayed variant is not mixed into the initial
comparison.

## Why total peak memory is not enough

For this model, unquantized KV storage is about 32 KiB per cached token, or
about 512 MiB at 16,384 tokens. Model weights and transient prefill buffers are
not compressed by KV-cache quantization. The experiment therefore records both:

- direct retained cache bytes from every MLX cache layer; and
- whole-application MLX active and peak memory.

Process peak RSS is a lifetime high-water mark and is reported only as context,
not used as the primary decision metric.

## Workloads

### Performance and memory

Use the existing five task types—Q&A, rewrite, context question, extraction,
and summarization—at 4K, 8K, and 16K. Each arm receives identical token IDs and
generation settings. Each request uses a fresh cache.

The full capture is one warmup plus ten measured repetitions per task and
length. Before that expensive run, use a bounded smoke test and a three-repeat
screen; only viable candidates advance.

### Long-context quality

Use fixed greedy fixtures with required facts placed near the beginning,
middle, and end of the context. Include exact fact retrieval, conditional fact
retrieval, structured extraction, grounded summarization, and multi-turn-like
history recall. Report each fixture and length separately rather than hiding a
failure in a pooled average.

## Captured metrics

- cache type, bit width, group size, quantization start, layer count, and layer
  offsets;
- direct sum of retained `cache.nbytes` and bytes per cached token;
- count/fraction of cache layers actually quantized;
- MLX active and peak memory, plus absolute MiB saved;
- TTFT, prompt throughput, decode throughput, and request duration;
- median and nearest-rank p95 per prompt and pooled by context length;
- output text/hash, generated-token count, and stop reason;
- required-fact and formatting checks for the quality suite.

## Approved success rule

### Correctness and quality gates

- Every 4K, 8K, and 16K case completes without crash, OOM, NaN, corrupted
  cache state, or incorrect cache length.
- No new critical omission, contradiction, or required-format failure versus
  the unquantized control.
- Aggregate required-fact score remains at least 95% of control.

### Memory gates at 16K

- 8-bit must reduce direct KV-cache bytes by at least 40%.
- 4-bit must reduce direct KV-cache bytes by at least 65%.
- Whole-application MLX peak memory must decrease by at least 10%; report the
  absolute MiB saved as well.

### Latency gates at every tested length

- Median TTFT regression is no more than 10%.
- Median decode-throughput regression is no more than 10%.
- Pooled p95 latency regression is no more than 15%.

### Adoption labels

- **Default for ordinary chat:** additionally no more than 3% slower at 4K and
  at least 5% total-memory or roughly 75–100 MiB savings there.
- **Long-context option:** passes the main 16K rule but is not worthwhile at 4K.
- **Research-only:** proves cache compression but misses adoption thresholds.
- **Reject:** loses a required fact, is unstable, or exceeds 20% slowdown.

Any selected winner must reproduce in a second full capture.

## Execution checklist

- [x] Implement direct cache-byte and quantized-layer measurement.
- [x] Validate cache offsets/types with one short smoke capture per arm.
- [x] Run fixed long-context quality fixtures for viable arms.
- [x] Apply the three-repeat screen gate (no candidate advanced because every
  arm had already failed a hard quality or peak-memory gate).
- [x] Apply the full-capture gate (no candidate qualified).
- [x] Apply the confirmation gate (not applicable; no selected candidate).
- [x] Write the decision, limitations, and raw-result paths.
- [x] Security-scan staged files before committing and pushing.

# Project context and current status

Last updated: 2026-10-05

## Objective

Practice the inference-engineering workflow used for production language-model
systems by measuring and optimizing one concrete local deployment:

- Apple M2 Pro MacBook Pro
- 16 GB unified memory
- `mlx-community/Llama-3.2-1B-Instruct-4bit`
- MLX 0.32.2 and MLX-LM 0.31.3
- single-user, concurrency-one interactive chat

The project optimizes inference behavior after training. It does not perform
pretraining, post-training, fine-tuning, or weight updates.

## Current selected configuration

- Model weights: published affine 4-bit, group size 64.
- Prompt-prefill step size: 2,048 tokens.
- Sampling for controlled tests: greedy.
- Sequential chat: reuse an exact token-prefix KV cache when the cache lifecycle
  and invalidation rules are satisfied.
- Fresh unrelated requests: use a fresh cache.
- Explicit compiled decode: disabled.
- KV-cache quantization: disabled.
- Speculative decoding with the Mixed-3/6 full-1B draft: disabled.
- Custom Metal kernels: none; profiling has not yet justified one.

These are benchmark-backed selections. The repository does not yet contain a
production chat server or application that integrates them as a deployment.

## Measurement contract

The accepted diverse baseline is
`benchmarks/results/diverse-baseline-20260925T053031Z.json`. It contains five
prompt types at target input lengths 64, 512, 2,048, and 4,096, with one warmup
and ten measured repetitions per fixture. The earlier single-prompt baseline is
retained but is not an equivalent workload.

Unless an experiment explicitly defines a different workload:

- hold model files, package versions, prompt construction, sampling, output
  cap, concurrency, and machine constant;
- change one inference variable at a time;
- preserve raw JSON and write an adjacent interpretation;
- report median and nearest-rank p95, actual token counts, TTFT, request time,
  prompt/decode throughput, memory, and output correctness;
- treat p95 from ten samples as a coarse tail indicator;
- run clean latency measurements without telemetry;
- use `powermetrics` or Metal capture only in separate diagnostic runs;
- do not promote a microbenchmark win without an end-to-end win.

The reusable procedure lives in `.agents/skills/mlx-inference-benchmark/`.

## Experiment record

### Baseline

The original and expanded baselines established latency, throughput, memory,
output, and environment measurements. The diverse baseline is the stable
reference for future comparable changes.

### Experiment 1: prompt-prefill step size

Tested 512, 1,024, 2,048, and 4,096. Keep 2,048: it had the strongest
long-prompt prefill throughput/TTFT balance. Step size 512 remains a
memory-oriented option. Step size 4,096 was slower and showed severe tail
outliers. Telemetry remained nominal but the primary timing interpretation is
limited because those captures ran with `powermetrics` and in ascending order.

Evidence: `benchmarks/results/prefill-step-size-analysis.md`.

### Experiment 2: cross-turn KV-prefix reuse

Adopt as a performance candidate for sequential chat. In the confirmation
pair, median cache-hit TTFT improved 71.49%, session peak MLX memory decreased
6.79%, all required facts passed, and the 200 measured turns happened to match
the control token-for-token. Exact equality is still diagnostic rather than a
general guarantee because finite-precision shape changes produced a benign
wording difference in an earlier run.

The optimization applies only when a new request has an exact cached token
prefix and cache state is isolated, bounded, and invalidated correctly.

Evidence: `benchmarks/results/experiment2-practical-analysis.md`.

### Experiment 3: explicit compiled decode

Reject the tested `mx.compile` boundary. Shapeless compilation could not infer
dynamic cache slices. Shape-specific compilation retraced once per token, made
the first token about 17 times slower, reduced subsequent decode throughput by
36.9%, and increased peak memory by 10.6% in the bounded probe.

Revisit only if MLX-LM provides a supported compiled decoder, cache state
becomes fixed-shape/dynamic-offset compatible, or profiling identifies a
smaller stateless compilation boundary.

Evidence: `benchmarks/results/experiment3-compiled-decode-analysis.md`.

### Experiment 4: weight quantization formats

Keep the existing published affine 4-bit/group-64 model. MXFP4 and Mixed-3/6
failed practical quality and did not achieve the required size reduction.
NVFP4 was slower at every screened input length and saved negligible memory.

Evidence: `benchmarks/results/experiment4-weight-quantization-analysis.md`.

### Experiment 5: KV-cache quantization

Reject immediate 8-bit/4-bit and post-prefill 8-bit/4-bit arms. Immediate
quantization caused large prefill memory and latency regressions. Moving
conversion to the prompt boundary reduced retained cache and active memory but
could not reduce the whole-request prefill peak.

The initial report incorrectly marked post-prefill 8-bit as passing quality.
Independent review found a new required-format failure at 8K: the control
produced the required three bullets and the candidate produced two. The test
also omitted the planned multi-turn history fixture, used an overly permissive
JSON substring evaluator, and named an aggregate check ratio
`required_fact_score`. These limitations are now explicit; they do not change
the overall rejection.

Evidence: `benchmarks/results/experiment5-kv-cache-quantization-analysis.md`
and `benchmarks/results/experiment5-review-corrections.md`.

### Experiment 6: speculative decoding

Reject the tested pairing. Draft counts one through four passed the exact-token
correctness gate, but every speculative arm was slower. The best real draft
arm still increased 128-token request time by 112.04%, reduced decode
throughput by 58.67%, and increased resident model memory by 85.5% before
request caches.

This does not reject speculative decoding generally. The draft retained the
same full 16-layer, 1B architecture and was only 14.5% smaller in stored
weights. A future test requires a genuinely small tokenizer-compatible draft
or a self-speculative/MTP/EAGLE-like mechanism that does not load a second
full-size model.

Evidence: `benchmarks/results/experiment6-speculative-decoding-analysis.md`.

### Experiment 7: runtime profiling

The phase-level profile is complete. With forced 128-token output, decode used
83.3% of request time at a 64-token prompt and 66.3% at 510 tokens. TTFT used
57.4% at 2,047 tokens and 69.3% at 4,096 tokens. Decode throughput also fell
from 216.2 to 135.6 tokens/s as prompt history grew.

Focused prefill and decode Metal traces were recorded, but this Mac has Command
Line Tools rather than full Xcode Metal Debugger. No GPU operation family has
therefore been assigned a trustworthy time share, and a custom Metal kernel is
not justified by current evidence.

Evidence: `benchmarks/results/experiment7-runtime-profile-analysis.md`.

## Known evidence limitations

- The baseline is specific to one Mac, OS/runtime combination, model, and local
  interactive workload; it does not establish performance on other Apple
  silicon or server GPUs.
- Some experiments used bounded screens after hard-gate failure rather than a
  full ten-repeat suite. Their speed values are elimination evidence, not
  precision estimates.
- Experiment 1 timing was telemetry-instrumented and ordered.
- Experiment 5 quality coverage and evaluator limitations are documented in
  its correction addendum.
- Experiment 7 identifies request phases but not dominant Metal kernels.
- Thermal pressure from `powermetrics` is categorical and was not useful as an
  internal-temperature measurement.

## Next research choices

1. Install/use full Xcode Metal Debugger to inspect the local Experiment 7
   traces, then pursue a custom kernel only if one operation family leaves a
   plausible path to at least a 10% end-to-end improvement.
2. If speculative decoding is revisited, begin with a purpose-built 100–300M
   tokenizer-compatible draft or self-speculative mechanism; repeat exact-token
   correctness before timing.
3. Extend the practical chat suite with actual multi-turn history cases and
   mechanically separate fact, format, contradiction, and structure gates.
4. Confirm any future winner with the full diverse benchmark and an independent
   repeat capture.

## Repository and privacy boundaries

Committed artifacts include source, plans, fixtures, sanitized telemetry,
result JSON, and analyses needed to understand or repeat the work.

The following remain local and ignored:

- `.venv/`;
- `models/`, including converted alternatives;
- partial or superseded captures listed in `.gitignore`;
- raw `.gputrace` directories, which are large and may contain local machine
  metadata.

Before every push, stage only intended files and run the local staged-content
security/privacy gate. Never commit credentials, administrator prompts,
personal process tables, absolute home paths, or raw machine captures.

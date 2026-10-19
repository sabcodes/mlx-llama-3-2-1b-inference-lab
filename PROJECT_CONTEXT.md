# Project context and current status

Last updated: 2026-10-19

## Experiment paper title

Experimental inference optimization for Llama on Apple Silicon, using MLX and Metal profiling

User-selected title, recorded 2026-10-19. Use this exact wording as the title
for the paper describing this project’s experiments.

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

Arno's dataset is from an Apple M4 Pro with 24 GB unified memory; the
local dataset is from an Apple M2 Pro with 16 GB unified memory. Keep these as
separate hardware cohorts. Do not pool their latency, throughput or memory,
interpret a cross-machine difference as an optimization gain, or treat M4
shader rankings as M2 measurements. M2 phase profiling is complete; its
operation-level GPU attribution remains pending. Completion of the combined
review does not complete that local measurement.

Diagnostic complete after reviewing Arno's `ArnoFeatures` commit
`1de18115d4f5323c07e4cac850feea1065ef3dee` on 2026-10-19. The original M2
phase profile and pinned M4 profile agree: decode dominates short prompts;
TTFT dominates long prompts. All four prompt hashes match, while the 510-token
output differs across devices despite stable outputs within each device.

Arno's Xcode 26.0.1 replay reports affine 4-bit matrix work as the largest M4
shader costs at a 2,047-token prompt: 90.93% matrix-matrix in prefill and 75.65%
matrix-vector in decode. These are manually transcribed replay shader-cost
shares, not clean request-time shares or M2 measurements. Independent replay,
absolute GPU durations, and a model-file hash manifest remain unavailable.

A focused matrix-path comparison is supported; custom Metal kernels remain
unselected until a concrete candidate preserves correctness and demonstrates
full-request improvement. No new local timing result was produced on October
19: the confirmation was stopped when the existing model was found offloaded
by iCloud. A local M2 GPU replay remains a follow-up.

Evidence: `benchmarks/results/experiment7-review-20261019.md`,
`benchmarks/results/experiment7-evidence-review-20261019.json`, and the original
M2/M4 analysis files. Specification and execution plan are in `experiments/`.

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
- Experiment 7 GPU ranking is reported M4 replay shader-cost evidence at one
  prompt length; no local independent replay or clean operation wall-time share
  is available. M2 GPU operation attribution remains unmeasured.
- Thermal pressure from `powermetrics` is categorical and was not useful as an
  internal-temperature measurement.

## Next research choices

1. Compare the reported 4-bit matrix bottleneck using existing MLX/runtime
   paths; require a concrete mechanism and a measured 10% full-request gain
   before selecting a custom-kernel implementation. Independently replay M2
   traces when compatible full Xcode is available.
2. Experiment 8 is a design-only M4 3B-target/1B-draft proposal in
   `research/experiment-8-and-followups-review.md`; no arms have run. It needs
   a fresh 3B baseline and corrected cross-model correctness checks. For the
   selected M2/1B deployment, revisit speculation only with a genuinely smaller
   compatible draft or a self-speculative mechanism.
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

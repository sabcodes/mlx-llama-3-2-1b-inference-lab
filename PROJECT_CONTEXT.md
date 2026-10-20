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

Diagnostic complete on 2026-10-20 under the documented CLI evidence contract.
Arno's M4 Pro / 24 GB and local M2 Pro / 16 GB results are separate cohorts;
never pool them or interpret hardware differences as an optimization gain.

The October 19 clean M2 confirmation completed 40 measurements and four warmups.
All prompt/output hashes match the original M2 run. Decode dominates short
prompts; TTFT dominates long prompts. Median request times are
0.868/1.131/2.197/3.776 s at 64/510/2,047/4,096 tokens. The OS changed and the
original power state is unknown, so cross-date timing shifts are observational.

New synchronized M2 recordings separately cover prompt processing plus first
output, and 32 new decode model calls after priming. Full 128-token outputs
match clean M2. Quantized matrix/dequantization contributes 91.53% of prefill
and 88.49% of decode summed shader Sample Duration. Sample unions, denominators,
clock conversion and independent GPU execution containment are validated.
These shares are not clean request fractions or exact dispatch timing.

The actual KVCache update path is isolated with evaluated synthetic inputs at
real model geometry. Five trials of 16 layers cover allocation, ordinary append
and growth; Active GPU interval unions across five trials are
4.949041/0.054209/5.855170 ms respectively. Synchronized wall medians are
6.819/3.741/13.505 ms per 16-layer trial. These metrics have different scopes;
no in-model request fraction is inferred. Ordinary append has no shader samples,
which does not imply zero GPU execution. Tiny/unsampled shader families remain
imprecise.

The original `.gputrace` files remain unreplayed. New validated CLI evidence
closes the diagnostic under the October 20 specification; independent replay
of the old captures is an optional replication, not a pending completion gate.
The first new recording set was excluded; the corrected worker synchronizes
both the dedicated generation stream and default stream, with collector idle
settling/drain outside the measured windows. All new traces retain 234,726,512
bytes, below the approved storage ceilings; raw traces remain local.

Arno's reviewed source remains `ArnoFeatures` commit
`1de18115d4f5323c07e4cac850feea1065ef3dee`. His M4 Xcode replay reports 90.93%
prefill matrix-matrix and 75.65% decode matrix-vector shader cost. These are
manually transcribed M4 replay costs, distinct from the new M2 Sample Duration
metric. One 510-token output differs across devices despite matching inputs.

Decision: examine the existing quantized matrix path first if pursuing another
optimization. Keep custom kernels unselected until a concrete correct candidate
demonstrates >=10% full-request improvement. No speedup was measured here.

Evidence: `benchmarks/results/experiment7-m2pro-phase-and-kv-analysis-20261020.md`,
`experiment7-m2pro-phase-validation-20261020.json`, final per-phase attribution
and workload/provenance JSON, plus the historical October 19 review/confirmation.
Specification and execution plan remain in `experiments/`.

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
- Experiment 7 uses sampled shader rankings and Active GPU interval unions;
  exact physical per-kernel times and clean request operation fractions are
  unavailable. Isolated synthetic KV costs are not in-model request fractions.
  Original M2 captures and Arno's M4 replay remain independently unreplayed.
- Thermal pressure from `powermetrics` is categorical and was not useful as an
  internal-temperature measurement.

## Next research choices

1. Compare the reported 4-bit matrix bottleneck using existing MLX/runtime
   paths; require a concrete mechanism and a measured 10% full-request gain
   before selecting a custom-kernel implementation. Independent replay of the
   original M2 captures is optional replication; the diagnostic is complete.
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

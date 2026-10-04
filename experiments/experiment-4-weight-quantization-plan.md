# Experiment 4: MLX weight quantization and kernel formats

## Plain-language question

Can the same Llama 3.2 1B Instruct model be stored and executed in a different
weight format that makes interactive chat faster or smaller on this M2 Pro,
without making its answers materially worse?

This is inference optimization. It does not train or fine-tune the model.

## Confirmed project constraints

- Hardware: the current Apple M2 Pro MacBook Pro with 16 GB unified memory.
- Runtime: the project-local `.venv`, MLX 0.32.2, and MLX LM 0.31.3.
- Model family: Llama 3.2 1B Instruct; no model larger than 1B.
- Workload: single-user interactive chat.
- Selected Experiment 1 setting: `prefill_step_size=2048`.
- Benchmark discipline: one warmup plus ten measured repetitions, greedy
  generation, a 128-token output cap, and the same five prompt types at 64,
  512, 2,048, and 4,096 input tokens.
- Timing runs are clean runs. `powermetrics` is optional and separate, not part
  of the primary decision.
- Model artifacts remain under `models/` and are excluded from Git.

## Fairness rule

All newly quantized candidates must be converted from the same pinned BF16
source checkpoint. We will not quantize the existing 4-bit weights again.
Re-quantizing already rounded weights would introduce a second source of
quality loss and would not be a fair comparison of quantization formats.

The existing `mlx-community/Llama-3.2-1B-Instruct-4bit` model remains the
deployed control. A fresh affine 4-bit conversion from the BF16 source is also
needed as a conversion control: it tells us whether differences are caused by
the current converter/runtime rather than by the candidate format itself.

## Candidate funnel

The installed MLX LM converter exposes affine integer quantization plus
`mxfp4`, `nvfp4`, and `mxfp8`, and mixed affine recipes. Support in a CLI is not
proof that a format is fast on this Mac, so each format must pass the same
screen.

### Stage A: reference and compatibility controls

1. Existing published affine 4-bit, group-size 64 model (current control).
2. Pinned BF16 source checkpoint (quality reference, not expected to win size).
3. Fresh affine 4-bit, group-size 64 conversion (conversion control).

### Stage B: initial speed/size candidates

1. `mxfp4` (fixed 4-bit/group-size-32 format).
2. `nvfp4` (fixed 4-bit/group-size-16 format).
3. `mixed_3_6` affine recipe at group size 64, which keeps selected sensitive
   projections at 6 bits while using 3 bits elsewhere.

An affine 6-bit or `mxfp8` candidate is a follow-up quality reference only if
the 4-bit candidates show unacceptable quality loss. A plain 2-bit candidate
is excluded from the first pass because its quality risk is high and the goal
is a useful chat model, not the smallest possible file.

## Measurements

For every candidate, capture:

- exact source revision and conversion command;
- quantization mode, bit width/recipe, and group size;
- configuration and weight hashes plus on-disk weight bytes;
- model load time and cold process-to-first-token time;
- time to first token (TTFT), prefill tokens/second, decode tokens/second, and
  total request time;
- MLX peak memory and process peak RSS;
- generated token count, stop reason, output text hash, and output text;
- per-prompt results plus median and nearest-rank p95 summaries.

## Quality checks

Exact token equality is recorded but is not the sole quality rule: different
quantized arithmetic can produce slightly different wording while preserving
the answer. The practical gate is:

- no missing required fact on the deterministic Q&A/context/extraction cases;
- valid requested structure for extraction cases;
- no refusal, corruption, repetition loop, or incoherent answer;
- disagreements with the control are reviewed against the BF16 answer;
- any candidate with a material correctness/usefulness loss is rejected even
  if it is faster.

## Execution checklist

- [x] Record repository revision, package versions, macOS version, and machine.
- [x] Run a fresh unchanged 4-bit control baseline with the fixed suite.
- [x] Pin and download one BF16 source checkpoint under ignored `models/`.
- [x] Record the BF16 source revision, sizes, and hashes.
- [x] Convert the fresh affine-4 control and initial candidates from that source.
- [x] Run load/generation and 512-token quality probes for each artifact.
- [x] Reject clearly incorrect candidates (`mixed_3_6` failed the context fact check).
- [x] Run a short screening benchmark on survivors.
- [x] Apply the full-suite gate (no candidate showed a plausible benefit, so no
  candidate advanced to a full ten-repeat run).
- [x] Compare quality first, then latency/throughput, memory, and disk size.
- [x] Apply the confirmation-rerun gate (not applicable because there was no
  qualifying candidate).
- [x] Write the decision, limitations, and raw-result paths.
- [ ] Scan staged files for secrets/private paths before committing and pushing.

## Predeclared decision rule

The user approved this rule as a whole on 2026-10-03.

A candidate is a practical winner only when it passes the quality gate and
does at least one of the following against the fresh affine-4 control:

- improves pooled median decode throughput by at least 10%; or
- reduces both model weight bytes and pooled peak MLX memory by at least 20%.

It must not regress pooled median TTFT by more than 10% or increase pooled peak
MLX memory by more than 10%, unless we explicitly classify it as a
higher-quality rather than a speed/size candidate. Results will also be shown
per input length so a pooled median cannot hide a long-context regression.

If no candidate clears these thresholds, keeping the current affine 4-bit
model is a valid and useful result.

## Stop conditions

- Do not overwrite or mutate the existing model.
- Do not convert from an already quantized candidate.
- Stop a candidate after a failed compatibility/correctness smoke test.
- Stop the full sweep if the Mac is under unusual load or thermal pressure;
  restart clean timing later rather than mixing conditions.
- Do not publish or commit model weights, credentials, absolute home paths, or
  raw process listings.

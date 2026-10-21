# Experiment 8 (local M2): packed quantized projections

Date: 2026-10-20
Status: execution authorized 2026-10-20; amended before implementation or runs.

## Question and hypothesis

Can the existing Llama 3.2 1B 4-bit model answer short-prompt requests at least
10% faster on this M2 Pro by grouping projections that consume the same input,
without changing weight values, generated tokens or the selected memory budget?

Hypothesis: a larger existing MLX quantized matmul can replace several smaller
calls, reducing submission/dispatch overhead and potentially repeated input
reads. Whether MLX actually benefits from the changed matrix shape is unknown.
The plan makes no prediction that packing will improve GPU bandwidth, native
kernel count or full-request time. Rejection is a useful experiment outcome.

## Why this is the next local experiment

Experiment 7 measured quantized matrix/dequantization at 91.53% of summed
prefill shader Sample Duration and 88.49% of decode on local M2. These are
sample shares, not clean request fractions or an Amdahl speedup estimate.
The pinned Llama source separately computes Q/K/V from one attention input,
and gate/up from one MLP input. This exposes a concrete change using the
existing quantized API before writing custom Metal code.

This is the local project's Experiment 8. The historical M4 3B-target/1B-draft
proposal reviewed in `research/experiment-8-and-followups-review.md` is separate
reference material, not an adopted local experiment or a prerequisite.

## Fixed deployment and invariants

- M2 Pro / 16 GB, same macOS/power mode/charger state within each comparison.
- Existing published Llama 3.2 1B Instruct affine-4/group-64 checkpoint.
- Project environment: MLX 0.32.2, MLX-LM 0.31.3. No dependency upgrade.
- Fresh KV cache per primary request, greedy, concurrency one, prefill step 2,048.
- Reuse baseline fixture builders: five tasks at targets 64/512/2,048/4,096.
- Pin template date to `04 Oct 2026` for every control/candidate fixture.
- Preserve tokenizer, RoPE, attention, normalization, cache implementation,
  sampling/log-probability path and tied embedding/output head.
- No training, re-quantization, activation quantization, new model, second model,
  custom kernel, compiled decode or change to speculative decoding.
- No request-time concatenation or alternate path selected using expected output.

## Candidate mechanism and arms

Concatenate output rows of already packed `weight`, `scales` and affine
quantization `biases`, without unpacking/re-rounding their values. Group size,
bits, mode, transpose orientation and tensor dtype must match before packing.
Use the installed `mx.quantized_matmul`; split the result at the exact original
output dimensions before the existing downstream operations.

| Arm | Change | Logical projection calls per layer/model forward |
|---|---|---:|
| C | Unchanged control | 7 |
| Q | Q/K/V packed; MLP unchanged | 5 |
| G | Gate/up packed; attention unchanged | 6 |
| QG | Both packings | 4 |

For this checkpoint, Q/K/V outputs are 2,048/512/512; the packed output is
3,072. Gate/up outputs are 8,192/8,192; the packed output is 16,384. Output and
down projections remain separate. Counts exclude the tied output head and are
logical source-level calls, not verified Metal dispatch counts or proportional
speedup claims. All three candidate arms are independently eligible for the
same bounded screen; QG is a declared combination, not a post-hoc new arm.

Packing occurs once after load, before warmup. Prefer one layer at a time;
materialize packed tensors and release obsolete original projection references.
Verify row/split boundaries and every packed component against its source slice.
Do not retain a hidden full copy of the original projections in adapters or
closures. Persist only the original checkpoint; do not save a converted model.
The ordinary linear bias is distinct from quantization `biases`: assert the
pinned model's Q/K/V and gate/up projections have no ordinary bias, otherwise
stop for a specification amendment rather than silently dropping it.

## Stage 0: correctness before speed

Record model/config/tokenizer and source hashes, versions, device, power and
fixture token hashes. Confirm actual dimensions, quantization attributes and
tied output-head identity. Stop on incompatibility; do not install a substitute.

1. Packed weight/scales/biases slices must be bitwise identical to the source.
   Check all 16 layers, row ordering, split shapes, strides and contiguity.
2. Compare separate versus packed projections on zeros, ones, seeded bounded
   random inputs and captured model activations. Include single-token decode
   and prefill row counts around dispatch/chunk boundaries: 1/2/31/32/33,
   63/64/65, 2,046/2,047/2,048/2,049 and 4,095. Also test 9/10/11,
   12/13 and 17/18/19 around pinned quantized dispatch transitions. No nonfinite values; diagnostic
   allclose tolerance is atol=1e-2, rtol=1e-2. Retain maximum/RMS differences.
   This tolerance never excuses a token mismatch.
3. Compare complete greedy token IDs, output lengths, EOS position and finish
   reason against a fresh same-process-version control for all 20 fixtures
   with normal EOS/cap 128, plus forced-output 128-token cases at four lengths
   and a forced 256-token 2,048-target case. Check cache offsets/layer shapes.
4. Reuse all scenarios from `experiment2-sessions.json` as an additional cache
   lifecycle correctness test, comparing control/candidate under identical
   fresh-history and exact-prefix-reuse policies. Hash each rendered history.
   This does not mix cache reuse into the primary fresh-cache speed benchmark.

Any packed-byte mismatch, numeric failure, token mismatch, invalid cache, crash
or new required-fact/format failure against a passing control rejects the affected
arm before speed selection. Stop at the first hard failure and retain evidence;
subsequent gates for that arm are explicitly skipped.
Check required facts, complete JSON structure and requested item counts when
reusing quality fixtures; do not repeat Experiment 5's permissive substring
checks. If control itself fails a fixture, document that shared limitation separately;
exact output identity establishes preservation, not task-quality correctness. Numerical differences or failures may motivate a new design,
but do not loosen gates after inspecting an arm's performance.

## Stage 1: bounded mechanism and request screen

- First inspect representative packed calls for extra copies, hidden original
  weight retention and changed kernel selection. A layer microbenchmark is
  diagnostic only; fewer Python calls do not prove fewer GPU dispatches.
- In separate clean processes, screen C/Q/G/QG on the summarization fixture at
  all four lengths, forced 128 output tokens, one warmup plus five repetitions.
  At most 80 measured requests and 16 warmups if every arm passes Stage 0.
- Use one recorded seeded arm order and identical fixture order. No concurrent
  inference or profiler. Report TTFT, decode, request, token identity and memory.
- Retain a candidate only if its median full-request time improves >=5% at at
  least one of the 64/512 targets, with <=5% median request/TTFT regression at
  every screened length and no memory/correctness failure.
- Select at most one finalist: greatest mean of the two short-length median
  request reductions, then lower worst-case long-length regression, then fewer
  packed projection families. Stop if none survives. A microbenchmark-only
  win, or only a long-prompt win, does not select a default under this question.

## Stage 2: full workload and independent confirmation

Define `exp8-packed-diverse-v1`: five tasks x four lengths, normal EOS, output
cap 128, fixed template date, one warmup and ten measured repetitions per
fixture. Each arm/run has 20 warmups and 200 measurements with completion
markers. The forced-output screen is `exp8-packed-screen-v1`; never pool it
with this workload. Historical benchmark captures are context, not the speed
or correctness denominator: acquire a new unchanged control on the same setup.

Run C then the chosen finalist in fresh processes. Independently repeat in
fresh processes with finalist then C. Record fixture/order seeds and identical
rendered token hashes within each pair. Keep both repeats separate and require
all gates in each; do not average a failed repeat away. Run clean timings with
no Instruments, Metal capture or powermetrics. Explicitly synchronize the
MLX-LM generation stream at request completion and use the same timing boundary
for control and candidate. That boundary is part of the new workload version;
do not silently repair or overwrite historical result files.

Promotion requires all of the following in both captures:

| Gate | Required result against its matched control |
|---|---|
| Primary benefit | >=10% lower pooled median full-request time at BOTH 64 and 512 target lengths |
| Per-fixture protection | No fixture's median request time or TTFT regresses >5% |
| Long prompts | Pooled median request time and TTFT regress <=5% at 2,048 and 4,096 |
| Tail protection | Pooled-by-length nearest-rank p95 request and TTFT regress <=10% at every length |
| Correctness | Exact token/length/EOS/finish agreement and all Stage 0 gates retained |
| Request memory | Per-length median request MLX peak <=105% of matched control; reset/request boundaries identical |
| Packing/startup memory | Extra transient load/packing peak <=512 MiB above control; no retained duplicate projections |
| Capacity | No swap growth during the arm; stop if process RSS exceeds 8 GiB on this 16 GB Mac |

Record both pooled-by-length and per-fixture summaries. Pooled-by-length has
50 observations; per-fixture nearest-rank p95 from ten is a coarse maximum.
If a tail gate remains ambiguous or repeats disagree, classify inconclusive
and propose a new predeclared confirmation, rather than repeatedly running
until it passes. Record process peak RSS as a process-lifetime high-water mark,
not a per-request delta. Measure cold load, packing duration and first request
separately; warm speed cannot hide repeated request-time setup or memory cost.

## Decision, deliverables and execution boundary

Outcomes: reject; inconclusive; or select a specific verified arm for the tested
fresh-cache workload. Any selected arm still needs a separate integration change
and cache-reuse performance validation before claiming production deployment.
No benchmark win or runtime default is declared by this proposal.

Future implementation should add an experiment-local adapter and thin runner,
reusing existing fixture/hash/measurement helpers. Do not edit site-packages.
Save unique partial/final result paths, completion markers, manifests, packing
checks, correctness failures, screen selection and independent confirmation.
Update analysis, current context, Obsidian task/board/daily/decision/evidence
notes and GitHub after the staged-content gate. Preserve rejected results.

The original proposal was design-only. On 2026-10-20 the user authorized
implementation and execution following the refreshed ordered plan. Retained
outputs remain within the proposed 10 MB timing/check/log allowance.
Execution is expected to need no retained model/runtime downloads or model
exports. A naive combined packing can temporarily duplicate about 342 MiB of
quantized parameters across 16 layers; layerwise construction should reduce
that peak, but actual allocation/cache behavior is unmeasured. Proposed timing
JSON/log output allowance is <=10 MB in `benchmarks/results/`; raw GPU captures,
new dependencies, larger outputs or external storage require a separate budget.
Check iCloud availability and existing storage before future execution.

## Evidence and primary sources

Checked 2026-10-20; installed pinned source governs execution, not current docs:

- [Local M2 diagnostic](../benchmarks/results/experiment7-m2pro-phase-and-kv-analysis-20261020.md).
- [Pinned MLX-LM v0.31.3 Llama projections](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/llama.py).
- [Pinned MLX v0.32.2 quantized linear path](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/quantized.py).
- [Quantized matmul API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html): current documentation is 0.32.4; confirm against installed 0.32.2 stubs before implementing.
- Model dimensions and quantization are pinned by the existing `models/Llama-3.2-1B-Instruct-4bit/config.json` and Experiment 7 source/model manifest; these local artifacts are not newly downloaded.

## Execution measurement clarifications (frozen before implementation)

Pinned generator source has one-token asynchronous lookahead. Synchronize
`generation_stream` and the default stream before request start and at request
end, including early-EOS closure; record end before post-request cache inspection.
Never retain response logprobs across tokens or requests. Record emitted response
token IDs including terminal EOS, terminal finish reason, actual emitted response
count, and visible non-EOS count separately; the API generation count includes
terminal EOS. Do not confuse them with evaluated lookahead model calls.

Record KV offsets, shapes, types and state/token signatures as correctness
evidence; compare arms under the same policy, not fresh versus reused caches
against one another. Prefix reuse must trim the cache to the common exact
rendered prefix and pass only its uncached suffix. Keep history construction
and assistant responses identical to matched control.

Record `pmset` power conditions, OS/device, swap bytes and process RSS at
process/case boundaries. Compare low-power setting and external supply state,
not fluctuating battery percentage. A changed power mode invalidates timing;
swap growth or process RSS >8 GiB stops the arm. Unavailable swap/RSS data
blocks promotion rather than being inferred as zero. Packing peak and retained
parameter bytes are independent of request-reset MLX peaks; track both.

Screen arm order is generated once using seed 20261020 and stored before
measurements. Stop each arm after a hard correctness failure; stage-2 remains
conditional on a surviving finalist. Cold/startup metrics are separate from
warm measurements. Retain a frozen plan/spec SHA-256 alongside source hashes.
No gate is relaxed after observing output or performance.

### Final preimplementation numerical and disposition rules

Numerical inputs use the embedding activation dtype, zeros, ones, uniform
[-1,1] with seed 20261020, and real normalized attention/MLP activations
from the fixed 2,048-target summarization fixture. Retain <=65 rows per layer
and tile those for larger diagnostic shapes. Evaluate one input/shape/family
at a time; save scalar max/RMS differences and first failure. Fixed final
selection tie-break is Q, then G, then QG. A failed finalist ends confirmation.

Quality validators: extraction must parse as one complete JSON object
(optionally a single complete fenced JSON block), never a substring;
summarization must contain exactly three nonempty bullet lines; history fact
checks use fixture alternatives, with observed control wording limitations
recorded separately. Any shared control failure leaves absolute quality
unpassed and blocks selection; exact tokens alone establish equivalence.
A ten-second idle interval precedes each timing process; record OS thermal
status where available and reverse fixture order for confirmation pair two.
Packing gate compares MLX peak bytes (extra <=512 MiB); also disclose RSS
high-water and cached allocator bytes without confusing them with active
allocation. Process RSS hard limit is 8*2**30 bytes.

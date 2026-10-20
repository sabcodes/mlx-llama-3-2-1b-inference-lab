# Experiment 7: M2 Pro confirmation and CLI profiling follow-up

Current completion update (2026-10-20): see
[separate M2 phase and isolated KV diagnostic](experiment7-m2pro-phase-and-kv-analysis-20261020.md).
Pending states below describe the historical checkpoint.


Date: 2026-10-19
Status: clean M2 confirmation and bounded CLI shader attribution validated.
Original separate-phase M2 GPU attribution remains pending.

## Hardware and measurement scope

These are new **Apple M2 Pro / 16 GB** measurements. Arno's **M4 Pro / 24 GB**
results remain a separate dataset; no M4 shader percentages appear in the
measurements below. The earlier cross-machine review is not a substitute for
local M2 operation attribution.

The existing model and `.venv` were kept downloaded through Finder. Metadata
verified zero dataless environment files before the successful run. No model,
package, environment or alternate checkout was installed or downloaded by the
benchmark. The pinned environment remains MLX 0.32.2 and MLX-LM 0.31.3.

## Clean confirmation setup

- Published Llama 3.2 1B Instruct affine-4/group-64 weights.
- Fresh cache, greedy generation, concurrency one, 2,048-token prefill steps.
- Fixed chat-template date `04 Oct 2026`; EOS suppressed, 128 output tokens.
- One warmup and ten measured requests per length: 40 measurements total.
- No powermetrics, Metal capture or xctrace process during clean timing.
- New macOS 27.0.1; Xcode 27.1 setup passed. The OS differs from
  the original M2 capture. Power source: AC power at both start and finish.

## Results

Medians except the explicit nearest-rank p95 column:

| Prompt tokens | TTFT (s) | TTFT share | Decode (s) | Decode share | Request (s) | Request p95 (s) | Decode tok/s | MLX peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.148 | 17.5% | 0.701 | 82.5% | 0.868 | 0.959 | 181.1 | 0.836 |
| 510 | 0.422 | 37.2% | 0.708 | 62.8% | 1.131 | 1.208 | 179.3 | 1.229 |
| 2,047 | 1.381 | 62.8% | 0.816 | 37.2% | 2.197 | 2.255 | 155.6 | 1.542 |
| 4,096 | 2.790 | 74.0% | 0.986 | 26.0% | 3.776 | 3.863 | 128.9 | 1.635 |

TTFT includes prompt processing, first-token work and sampling. Decode covers
127 transitions after the first output token. Shares are medians of per-run
ratios; independently summarized component medians need not sum exactly to the
median request time. With ten measurements, nearest-rank p95 is the maximum.

The phase crossover repeats: decode dominates the 64- and 510-token inputs,
while TTFT dominates the 2,047- and 4,096-token inputs. Longer histories also
reduce subsequent decode throughput, from 181.1 to 128.9 tokens/s in this run.

## Validation and comparison limits

All four prompt hashes **and all four output hashes** match the original M2
profile. Every measured request generated 128 tokens; outputs were stable
within each case. Independent recomputation of medians, p95s, minima/maxima,
timing identities, phase shares and throughput denominators passed.

Current median request times are higher than the original October 4 values
(0.707/0.947/1.801/3.056 s). Do not attribute that difference to a code regression
or optimization: the OS changed and the original power source was unrecorded.
The short-prompt p95 also shows more variation than the original capture.
This confirmation supports the phase split and tested output identity, not a
controlled causal estimate across dates. No deployment setting is changed.

## CLI profiling checkpoint

Full Xcode is now installed and first-launch setup passes. The supported
`xctrace export` CLI rejects the existing `.gputrace` format with exit code 11.
Metal System Trace and Metal GPU Counters are available for new recordings.

The launch-based pilot had no Python GPU rows and is excluded. The first warmed
attach attempt did not finalize; the held-target retry was interrupted by a
storage guard and is malformed. Neither supports operation attribution.
The final held-target retry completed and exported successfully with Xcode
27.1 (27A9275). It retained 137,245,510 bytes; all pilot recordings together
retain 247,936,324 bytes, with observed combined peak contents 383,709,630 bytes.
These are logical file bytes, not a bound on Xcode caches or total temporary use.
No new recording was needed when resuming the interrupted chat.

### M2 sampled shader attribution

The final recording's actual duration is 5.996984 seconds for a requested
five-second window. It contains 14,988 target shader timeline rows; 166 rows
from other processes are excluded. The worker completed three 2,047-token,
fresh-cache requests with 128 output tokens each, matching the clean M2 output
hash. Their combined wall time exceeds the trace window, so the third request
is only partly captured. Prefill and decode are mixed, without phase markers.

Metric: Instruments `metal-shader-profiler-intervals` **Sample Duration**, raw
integer nanoseconds. The denominator is the sum of that field across target
Python rows: **4.045616334 seconds**. This is an instrumented sample-duration
ranking, not exact kernel dispatch time, M4 replay shader cost, or clean
request-time attribution.

| Family | Summed sample duration (s) | Share of target sample-duration sum |
|---|---:|---:|
| Quantized matrix + dequantization | 3.627792 | 89.67% |
| Attention | 0.334022 | 8.26% |
| Copy/indexing (unassigned) | 0.041682 | 1.03% |
| Elementwise/sampling/other | 0.039006 | 0.96% |
| RMS normalization | 0.003115 | 0.08% |

The largest labels are `affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0`
(2.582767393 summed seconds) and
`affine_qmv_fast_float16_t_gs_64_b_4_batch_0` (1.044899509 summed seconds).
Attention includes `steel_attention` and both `sdpa_vector_2pass` stages.
Copy/gather/scatter work is not enough to identify KV updates; their separate
cost remains unknown. Labels not sampled are unknown, not zero.

Overlap treatment: retain all sampled rows in the stated sum, then compute
interval-union coverage separately. The target union is **4.037414018 s**,
8.202316 ms below the sum. Family unions can overlap and must not be added or
interpreted as disjoint request fractions. Nested timeline depth is retained;
no unsupported parent/child removal is applied.

### M2-specific decision and remaining acceptance gate

Quantized matrix work dominates this bounded M2 sample-duration metric. This
supports examining the existing matrix path on M2 independently of Arno's M4
ranking. It establishes no custom-kernel gain and changes no selected runtime.
For an actual end-to-end fraction f and candidate speedup s, the ideal reduction
would be f*(1-1/s); this pilot does not measure f, so its 89.67% must not be used
as that fraction or converted to a claimed >=10% request improvement.

The approved feasibility pilot is complete. The original operation-level gate
is still pending: replay both original phase-specific captures, or obtain
separately scoped phase-specific M2 evidence with reliable boundaries and KV
attribution. The CLI rejects `.gputrace`; this mixed pilot does not replace
those captures. No larger profiling campaign is started under the pilot scope.

To regenerate sanitized evidence from the existing local recording (choose a
new output path):

```sh
python3 -B benchmarks/analyze_experiment7_trace.py \
  --trace benchmarks/results/experiment7-m2pro-cli-pilot-complete-20261019.trace \
  --output benchmarks/results/<new-run-id>.json
```

Apple documents the newer shader-cost graph as M3-or-later only. Any usable M2
counter or sampled-duration result must state its actual metric/denominator and
must not be relabeled as Arno's M4 replay shader-cost share. Source checked
2026-10-19: https://developer.apple.com/documentation/xcode/optimizing-gpu-performance

## Evidence

- Clean run: `experiment7-m2pro-phase-confirmation-20261019-r2.json`.
- Model/source hashes, OS and power: `experiment7-m2pro-confirmation-provenance-20261019-r2.json`.
- Independent validation: `experiment7-m2pro-confirmation-validation-20261019.json`.
- CLI capabilities: `experiment7-m2pro-cli-capabilities-20261019.json`.
- Validated pilot attribution: `experiment7-m2pro-cli-pilot-complete-validation-20261019.json`.
- Validated workload: `experiment7-m2pro-cli-pilot-workload-validated-20261019.json`.
- Pilot commands and outcomes: adjacent `experiment7-m2pro-cli-pilot-*provenance*.json`.
- Specification and current plan: `experiments/experiment-7-runtime-profiling-spec.md`
  and `experiments/experiment-7-runtime-profiling-plan.md`.

Raw `.trace` recordings and regenerated XML exports remain local and ignored.

# Experiment 7: completed M2 phase and isolated KV diagnostic

Date: 2026-10-20
Status: diagnostic complete under the explicitly bounded CLI evidence contract.
No custom-kernel optimization or speedup is established.

## Scope and correctness

Apple M2 Pro / 16 GB; published Llama 3.2 1B affine-4/group-64 weights;
MLX 0.32.2, MLX-LM 0.31.3, Xcode 27.1, macOS 27.0.1. The current model/config
hashes match the clean confirmation. Arno's M4 Pro / 24 GB evidence remains
separate. No model, environment, dependency or checkout was downloaded/installed.

The completed October 19 clean profile remains the primary phase-latency result:
40 measurements and four warmups. These new instrumented recordings explain
operations at a 2,047-token prompt; they are not optimization arms or a repeated
200-request diverse baseline. Greedy, EOS suppression, fresh cache and the
pinned `04 Oct 2026` prompt date are preserved.

The unchanged `generate_step` is paused at synchronized boundaries. Prefill
includes all prompt processing and first-output sampling, then stops before
the first decode lookahead model call: all 16 cache offsets are 2,047. Decode
primes four yielded outputs and one evaluated lookahead outside the recording,
then executes exactly 32 new single-token model calls: offsets 2,051 to 2,083.
Both workers finish 128 outputs after recording and match the clean M2 output
hash `57adb9401e384308c61e6a5ac3d4f2ada687ce0a7e07cd198017ca453027059d`.

## Phase GPU evidence

Instruments `metal-shader-profiler-intervals` Sample Duration is an instrumented
timeline metric. Its denominator is the sum over target-worker shader rows in
the named phase window, in raw integer nanoseconds. Overlaps remain in the sum;
interval-union coverage is reported independently. Family unions are not
additive. One-nanosecond entries are retained verbatim and do not establish
precise physical kernel costs.

Worker Mach timestamps are converted through the recording's `time-info`
epoch/timebase and used to clip sampled intervals. Independent Active GPU
execution timestamps are fully contained in each worker span. The idle
collector settling/drain periods and other processes are excluded.

| Phase | Shader rows | Summed sample duration (s) | Sample interval union (s) | Active GPU execution union (s) |
|---|---:|---:|---:|---:|
| Prefill | 256 | 0.921613018 | 0.921306967 | 0.942018246 |
| Decode | 1,994 | 0.142009879 | 0.142003656 | 0.202508668 |

| Operation family | Prefill share of sample-duration sum | Decode share of sample-duration sum |
|---|---:|---:|
| Quantized matrix/dequantization | 91.52844% | 88.48565% |
| Attention | 7.02283% | 6.52558% |
| Copy/indexing, not attributed to KV in these phase traces | 0.04195% | 4.93374% |
| Elementwise/sampling/other | 1.33285% | 0.05501% |
| RMS normalization | 0.07394% | 0.00002% |

Quantized matrix/dequantization dominates this metric in both phases. Attention
is the next largest sampled family. The decode normalization result consists
of 31 one-nanosecond rows; its precise cost is unresolved, not demonstrated
negligible. Sample coverage is incomplete: 0.142003656 s of sampled union versus
0.202508668 s of Active GPU union in decode. No sampled percentage is a clean
request fraction. Matrix-matrix versus matrix-vector labels and the complete
per-label raw sums are preserved in the attribution JSON.

## Source-attributed isolated KV costs

The pinned `mlx_lm.models.cache.KVCache.update_and_fetch` path is invoked directly
with evaluated synthetic fp16 inputs and actual model geometry: 16 layers,
8 KV heads, head dimension 64. No projection, RoPE or attention work occurs in
these windows. Inputs and cache storage are materialized before recording;
all numerical checks run after recording and verify appended values and
unchanged prefixes.

Three paths are covered: initial allocation with 2,046 input tokens; one-token
append at offset 2,051/capacity 2,304; and one-token growth at offset/capacity
2,048. Each has five trials of 16 cache updates. The synchronized wall median
is per 16-layer trial and includes Python submission/evaluation/synchronization.
The GPU columns cover **all five trials**, not a per-trial median. Active GPU
execution union uses target-only `metal-gpu-intervals` Active intervals across
all channels; overlap is removed. This source-isolated metric can attribute
KV execution even when individual shader samples are unavailable.

| KV path | Synchronized wall median per trial (ms) | Active GPU union, five trials (ms) | Shader sample sum, five trials (ms) |
|---|---:|---:|---:|
| initial_allocation | 6.819 | 4.949041 | 2.862334 |
| ordinary_append | 3.741 | 0.054209 | Unmeasured |
| growth_at_2048 | 13.505 | 5.855170 | 4.445500 |

Ordinary append has no shader samples but does have 0.054209 ms of Active GPU
execution across the five trials. It is not zero-cost. Allocation and growth
show greater isolated GPU work than append in these observations. These costs
cannot be divided by ordinary model phase/request time to estimate an in-model
KV fraction: synthetic inputs, synchronization, instrumentation and allocation
lifetime differ. Fused/in-model cache work remains unseparated at that level.

## Validation, rejected attempts and storage

The first recordings were preserved and excluded. Prefill had no target shader
samples, and decode execution exceeded its default-stream-only boundary by
about 6 ms. The pinned generator uses a dedicated `generation_stream`; the
corrected worker explicitly synchronizes it and the default stream. The
controlled retry also adds two seconds of idle collector settling and drain.
The exact reason for missing first-run samples is not established. Both fixes
are disclosed; no old clean timing result was retroactively changed.

The corrected three recordings saved with exit code zero. Source/controller
hashes, model identity, output identity, numerical KV checks, interval
containment, family totals and overlap identities passed. The analyzer has
runnable reference-resolution, target-isolation, clipping/empty-window and
union checks. Raw traces, initial/intermediate exports and partials remain local.

Accepted recordings retain 119,452,563 logical bytes. All new recordings,
including the excluded set, retain 234,726,512 bytes. Observed combined peak
contents were 254,133,611 bytes, below the approved 1 GB additional retained and
2 GB observed working-output ceilings. Xcode cache/unobserved temporary growth
remains unknown. Existing traces were preserved.

## Final decision and completion boundary

Experiment 7's M2 diagnostic is complete under the October 20 CLI contract:
clean phase confirmation, separately bounded operation rankings, isolated KV
update costs and a scoped kernel decision are recorded. The original
`.gputrace` captures remain unreplayed; equivalent new CLI evidence closes the
diagnostic, not a claim of reproducing those old captures.

Investigate the existing quantized matrix path first if pursuing another
optimization. Keep custom kernels unselected until a concrete candidate
preserves correctness and demonstrates >=10% full-request latency reduction.
For an actual end-to-end fraction f and operation speedup s, the idealized
reduction is f*(1-1/s). These shader shares do not measure f; no numerical
request-speedup prediction is justified. No selected deployment setting changes.
Experiment 8 remains design-only and outside this execution.

## Reproduction and evidence

With the existing downloaded environment/model and a separately authorized
recording budget, choose a new ID:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -B benchmarks/record_experiment7_phases.py --run-id <new-id>
python3 -B benchmarks/analyze_experiment7_trace.py \
  --trace benchmarks/results/experiment7-m2pro-phase-prefill-<new-id>.trace \
  --workload benchmarks/results/experiment7-m2pro-phase-prefill-<new-id>-workload.json \
  --output benchmarks/results/<new-attribution-id>.json
python3 -B benchmarks/analyze_experiment7_trace.py --self-test
```

Repeat analysis for decode and KV. Outputs must use new paths.

- Final artifacts: `experiment7-m2pro-phase-{prefill,decode,kv}-20261020-r2-{workload,provenance,attribution}.json`.
- Independent containment records: adjacent `*-containment.json`.
- Audit: `experiment7-m2pro-phase-validation-20261020.json`.
- Model/runtime/source identity: `experiment7-m2pro-phase-manifest-20261020-r2.json`.
- Historical clean timing: `experiment7-m2pro-confirmation-analysis-20261019.md`.
- Specification/plan: `experiments/experiment-7-runtime-profiling-{spec,plan}.md`.

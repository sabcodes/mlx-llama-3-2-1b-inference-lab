# Experiment 7: M4 Pro runtime profile

Date: 2026-10-05 PDT (2026-10-06 UTC)

## Decision

The fresh M4 Pro run establishes a reproducible request-phase split. Decode
dominates short prompts; time to first token (TTFT) dominates long prompts.
Xcode Metal Debugger replay identifies MLX's affine 4-bit matrix operations as
the largest shader cost in both phases: matrix-matrix work during prefill and
matrix-vector work during decode. This supports a focused bottleneck study,
but does not yet justify writing a custom kernel or claim an end-to-end gain.

## Setup and provenance

- Apple M4 Pro, 24 GiB unified memory, macOS 26.3.
- Published `mlx-community/Llama-3.2-1B-Instruct-4bit` checkpoint. Its weights,
  config, and tokenizer SHA-256 hashes match the earlier M2 experiment.
- Python 3.13.7, MLX 0.32.2, MLX-LM 0.31.3, Transformers 5.17.0.
- Greedy generation, EOS suppressed, 128 output tokens, one request at a time,
  fresh KV cache, 2,048-token prefill step size.
- One warmup and ten clean measured requests at each prompt length. Metal
  capture and `powermetrics` were absent during timing.
- The Llama chat template inserts the current date. This rerun fixes
  `date_string` to `04 Oct 2026`; all four prompt-token hashes now exactly
  match the earlier M2 record. The initial M4 timing file
  `experiment7-m4pro-phase-profile-20261006T054000Z.json` used the changing
  date and is diagnostic only, not the confirmed result.

## Confirmed M4 Pro phase results

Medians from the [timing record](experiment7-m4pro-phase-profile-pinned-20261006.json):

| Prompt tokens | TTFT | TTFT share | Decode time | Decode share | Request time | Decode tokens/s | Per-run p95 inter-token latency | MLX peak memory |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.081 s | 14.8% | 0.466 s | 85.2% | 0.546 s | 272.6 | 3.92 ms | 0.836 GB |
| 510 | 0.237 s | 32.4% | 0.496 s | 67.6% | 0.735 s | 256.3 | 4.17 ms | 1.229 GB |
| 2,047 | 0.822 s | 58.0% | 0.593 s | 42.0% | 1.410 s | 214.2 | 4.92 ms | 1.542 GB |
| 4,096 | 1.676 s | 70.8% | 0.693 s | 29.2% | 2.374 s | 183.2 | 5.51 ms | 1.635 GB |

TTFT includes prompt work, the first generated token, and sampling. Decode
time covers the remaining 127 streamed token transitions. A percentage of
request time is a phase attribution, not an operation-level GPU profile.

All 40 measured requests generated 128 tokens, and each case had one stable
output-token hash across its ten runs. Three of four output hashes also match
the earlier M2 run. The 510-token output differs across machines despite an
identical prompt hash and stable output on each machine. Thus the comparison
establishes matched input and length, not universal cross-device greedy token
identity.

The M4 Pro median requests are about 1.28–1.29 times faster than the earlier
M2 Pro medians at these matched prompt lengths. This is a machine comparison,
not an optimization-arm speedup. Decode throughput falls about 33% from 64 to
4,096 prompt tokens, so long context raises the cost of subsequent tokens too.

## Metal captures and GPU operation ranking

Two local `.gputrace` directories were recorded at a 2,047-token prompt:

- [Prefill capture metadata](experiment7-m4pro-prefill-capture-20261006.json):
  1.029 GB of trace contents, includes first-token work and possibly one
  pipelined decode operation.
- [Decode capture metadata](experiment7-m4pro-decode-capture-20261006.json):
  1.301 GB of trace contents, roughly 16 decode steps after prefill.

The raw traces are gitignored and remain on this Mac. Capture wall times
include recording overhead and are not benchmark results. Xcode 26.0.1 Metal
Debugger replayed both traces with **Profile after replay**. Its shader-cost
ranking was [transcribed here](experiment7-m4pro-xcode-gpu-profile-20261006.json):

| Capture | Largest shader costs in Xcode replay |
|---|---|
| Prefill / first token | `affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0` 90.93%; Steel attention 6.82% |
| Decode | `affine_qmv_fast_float16_t_gs_64_b_4_batch_0` 75.65%; two-pass attention 11.74% + 1.87%; `vn_copyfloat16float16` 8.50% |

These percentages are shares of Xcode's **profiled shader cost**, not shares
of clean request wall time. The captures each cover one 2,047-token prompt;
their boundaries are approximate due to MLX-LM's pipelined generation. They
also use replay and profiling conditions rather than the clean timing setup.

The large 4-bit matrix shares make these operations appropriate for a focused
comparison of the existing MLX implementation with other runtime or kernel
choices. They do not demonstrate that MLX's kernels are inefficient. A custom
kernel candidate needs an explicit mechanism, exact-output validation, and a
measured full-request improvement. Removing the decode copy shader entirely
would save at most its 8.5% **shader-cost** share in this one capture, before
accounting for other request work; it is not by itself a convincing 10% total
request target.

| Gate | Result |
|---|---|
| Matched deterministic prompts | Pass: four of four prompt hashes match M2 |
| Clean M4 timing and dominant request phases | Pass |
| Focused prefill and decode traces | Pass |
| Dominant GPU operation family | Pass: quantized matrix work in both phases |
| Evidence for a custom kernel implementation | Not yet: a faster replacement has not been demonstrated |

The next implementation decision should use a bounded, matched comparison of
the 4-bit matrix path (for example, an existing MLX option or another runtime)
and promote it only after exact-output and full-request gates pass. A custom
Metal kernel remains a later option if that comparison shows a plausible
mechanism that existing kernels do not supply.

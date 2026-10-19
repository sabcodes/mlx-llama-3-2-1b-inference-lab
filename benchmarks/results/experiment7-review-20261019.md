# Experiment 7 review and completion record

Date: 2026-10-19
Status: diagnostic complete with reported M4 GPU evidence; no custom-kernel win.

## Hardware scope

Arno's dataset is from an Apple M4 Pro with 24 GB unified memory; the
local dataset is from an Apple M2 Pro with 16 GB unified memory. Keep these as
separate hardware cohorts. Do not pool their latency, throughput or memory,
interpret a cross-machine difference as an optimization gain, or treat M4
shader rankings as M2 measurements. M2 phase profiling is complete; its
operation-level GPU attribution remains pending. Completion of the combined
review does not complete that local measurement.

## Outcome

The M2 and M4 profiles agree on the request-phase crossover: short prompts are
decode-dominated, while long prompts are dominated by time to first token
(TTFT). Arno's M4 Xcode replay reports affine 4-bit matrix operations as the
largest shader costs. A focused matrix-path comparison is a justified next
study. A custom kernel remains unselected because no faster candidate or
end-to-end improvement has been demonstrated.

Completion here means review and consolidation of the existing diagnostic
measurements under the amended specification. It does not mean this M2 Mac
replayed the M4 traces or measured a kernel replacement.

## Source and attribution

Incorporated the seven new evidence/research files and minimal prompt-date fix
from GitHub user [axrno](https://github.com/axrno), branch `ArnoFeatures`, pinned
[commit 1de18115d4f5323c07e4cac850feea1065ef3dee](https://github.com/sabcodes/apple-silicon-inference-engineering-llama-3-2-1b/commit/1de18115d4f5323c07e4cac850feea1065ef3dee).
Imported artifacts retain their source bytes. This review adds limitations and
validation without rewriting the source analysis. The local M2 deployment
configuration, original measurements, and user-selected paper title remain.

## Verification of saved measurements

Recomputed medians, nearest-rank p95s, minima and maxima for every summarized
metric across three profiles (120 measured requests, plus 12 warmups). Checked
completion markers, four cases per profile, ten repetitions and one warmup per
case, 128 output tokens, stable per-case output hashes, timing identities,
phase-share calculations, and throughput denominators. All checks passed.

Only the original M2 and **pinned** M4 records form the matched-input comparison:

| Actual prompt tokens | M2 Pro 16 GB median request | M2 dominant phase/share | M4 Pro 24 GB median request | M4 dominant phase/share |
|---:|---:|---|---:|---|
| 64 | 0.707 s | Decode 83.3% | 0.546 s | Decode 85.2% |
| 510 | 0.947 s | Decode 66.3% | 0.735 s | Decode 67.6% |
| 2,047 | 1.801 s | TTFT 57.4% | 1.410 s | TTFT 58.0% |
| 4,096 | 3.056 s | TTFT 69.3% | 2.374 s | TTFT 70.8% |

All four prompt-token hashes match. Output hashes match for three cases; the
510-token output differs between devices. Output is stable within each device's
ten measured runs. This supports phase comparison, not universal cross-device
token identity. These are different machines and dates, not optimization arms.

The initial, unpinned M4 record has different input hashes at every length.
Retain it as diagnostic evidence and exclude it from matched-input comparisons.
The prompt-date fix pins Experiment 7's chat template to `04 Oct 2026` and records
that choice. Existing callers retain their date behavior unless they opt in.

## GPU evidence and its limits

Arno's manually transcribed Xcode 26.0.1 **Profile after replay > Shaders > Cost**
record reports, for one 2,047-token prompt per phase:

| Phase | Largest reported shader | Replay shader-cost share |
|---|---|---:|
| Prefill/first token | Affine 4-bit matrix-matrix multiplication | 90.93% |
| Decode | Affine 4-bit matrix-vector multiplication | 75.65% |

Listed costs total 99.07% in prefill and 98.65% in decode, leaving 0.93% and
1.35% unlisted. Capture metadata agrees on device, prompt length, date pin, and
trace names. Boundaries include pipelined look-ahead work.

These shares are **not clean request-time shares**. Absolute GPU durations,
screenshots/exports and raw M4 traces are absent from the committed evidence;
this review cannot independently reproduce the transcript. Do not infer an
end-to-end speedup, transfer percentages to M2, or conclude the MLX kernels are
inefficient. The source analysis reports matching weight/config/tokenizer
SHA-256 hashes, but no such hash manifest is committed, so this review verifies
matching recorded weight bytes and prompt hashes only.

Both local M2 traces exist and their logical sizes match their saved metadata:
976,304,257 and 1,166,696,898 bytes. Size consistency does not establish trace
integrity or replayability. Full Xcode remains unavailable on this Mac.

## Local confirmation and storage

An October 19 confirmation attempt was stopped before producing any timing
result when the existing model weight file was found to be an iCloud `dataless`
placeholder. Its launch provenance is retained locally as an ignored partial.
No new weights, traces, environment, installation, or alternate checkout were
created. Before a future local rerun, use Finder **Keep Downloaded** for
`models/Llama-3.2-1B-Instruct-4bit/` and `.venv/` in this project. A local replay
also requires a compatible full Xcode installation. No upgrade is required to
review the already-committed M4 evidence.

## Experiment 8 status

`research/experiment-8-and-followups-review.md` is a **design proposal** for a
3B target / 1B draft study on M4 Pro. No Experiment 8 arm has run. It requires a
fresh 3B target-only baseline and corrected cross-model correctness checks;
it does not change the selected 1B/M2 deployment or authorize model downloads.

## Evidence and plan

- Specification: `experiments/experiment-7-runtime-profiling-spec.md`.
- Execution/amendment record: `experiments/experiment-7-runtime-profiling-plan.md`.
- Machine-readable audit: `experiment7-evidence-review-20261019.json`.
- Original M2 interpretation: `experiment7-runtime-profile-analysis.md`.
- Source M4 interpretation: `experiment7-m4pro-runtime-profile-analysis.md`.
- GPU transcript: `experiment7-m4pro-xcode-gpu-profile-20261006.json`.

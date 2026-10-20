# Experiment 7 specification: runtime profiling completion

Date: 2026-10-19
Status: diagnostic complete on 2026-10-20 under the CLI evidence contract below.
Separate M2 phase and isolated KV attribution are validated; original saved
.gputrace captures remain unreplayed.

## Objective and scope

Finish the runtime diagnosis for the selected Llama 3.2 1B affine-4/group-64
configuration on the existing M2 Pro. Identify dominant request phases and,
where tooling permits, dominant GPU operation families. Decide whether evidence
supports a custom-kernel follow-up with at least 10% lower end-to-end latency.
This experiment does not implement a kernel or change the selected runtime.

The original protocol remains `research/experiment-7-runtime-profiling.md`.
This specification makes the previously pending work and acceptance criteria
explicit; it does not retroactively redefine the October 4 measurements.

## Measurement contract

- Reuse the local model and environment; no model, dependency, or tool downloads.
- MLX 0.32.2, MLX-LM 0.31.3; record actual versions before confirmation.
- Greedy, concurrency one, fresh KV cache, 2,048-token prefill steps.
- Summarization at target prompt lengths 64/512/2,048/4,096; record actual lengths.
- Suppress EOS, force 128 output tokens, one warmup and ten measured runs per case.
- Clean phase timing excludes GPU capture and powermetrics.
- Preserve original JSON and traces. Use a unique path for confirmation results.
- The 40-run phase diagnostic differs from the diverse 200-run baseline; do not
  claim an optimization win or directly comparable full-baseline result.

## Required deliverables and acceptance criteria

1. **Evidence validation:** machine-readable audit verifies completion markers,
   four cases, warmup/repetition counts, generated lengths, token hashes, timing
   identities, and recomputed medians/p95s. Verify both saved trace sizes against
   their capture metadata without replaying or copying the traces.
2. **Independent phase confirmation:** run the unchanged profiling runner once,
   preserve the new JSON, compare prompt/output hashes, and report timing shifts
   and phase dominance separately for both dates. Cross-date timing differences
   are observational, not attributed to an optimization.
3. **GPU attribution:** inspect both existing captures with Xcode Metal Debugger.
   Save sanitized operation labels, durations, units, aggregation denominator,
   replay/tool version, trace identity, overlap treatment, and capture boundaries.
   Cover quantized matmul, attention, normalization/elementwise work, KV updates,
   and other/unattributed work. Counts or binary labels alone are not timings.
4. **Decision:** a dominant request phase has median share >=50%. For a candidate
   operation occupying end-to-end fraction f and assumed speedup s, the idealized
   request-time reduction is f*(1-1/s). State assumptions; this is an upper-bound
   model, not a measured win. Trace replay shares cannot be silently equated to
   clean request shares. An unmeasured family is unknown, not zero.
5. **Publication:** update analysis, README, project context, and the existing
   Obsidian project/experiment/decision/task/index/daily/operation notes. Run the
   staged-content gate and push only intended, sanitized repository artifacts.
   Private vault notes, raw traces, models, and the environment remain ignored.

## Completion rule

Phase confirmation and evidence auditing can complete with existing tools.
Full operation-level completion requires trustworthy GPU timing and the final
custom-kernel decision. If the debugger is unavailable, explicitly retain the
experiment as in progress with a tooling blocker. Do not interpret a missing
measurement as evidence against custom kernels or silently close the task.

## Storage and prerequisite boundary

Reuse the approximately 2.1 GB of existing trace contents. New text/JSON evidence
is expected to stay below 1 MB in this project. No new captures or installations
are authorized by this plan. Full Xcode is absent from standard application
locations as checked on 2026-10-19; installation size and temporary storage are
not yet established. Agree on that budget and location before installing it.

## Primary documentation

Checked 2026-10-19; installed runtime versions remain pinned:
- https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html
- https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically

## Specification amendment: cross-machine evidence consolidation

User-directed 2026-10-19 after discovering `ArnoFeatures` at commit
`1de18115d4f5323c07e4cac850feea1065ef3dee` by `axrno`.

The completion target now includes review and incorporation of already-completed
M4 diagnostics, rather than requiring a new M2 replay to acknowledge that work.
Accept a reported operation-family ranking when its Xcode version, replay mode,
shader-cost denominator, device, prompt size, capture boundary, and provenance
are explicit. Absolute GPU durations and independent replay are unavailable;
record that limitation rather than fabricating them. Do not convert shader-cost
percentages to clean request-time shares or transfer them to M2 hardware.

Acceptance requires: validated 40-run M2 and pinned 40-run M4 records; four
matching input hashes; correct disclosure of cross-device output differences;
separate treatment of the unpinned diagnostic M4 file; focused verification of
the prompt-date fix; a provenance-backed summary; Obsidian synchronization; and
verified GitHub publication. The diagnostic may close with a measured phase
split, reported M4 shader ranking, and a decision to defer custom kernels until
a concrete candidate demonstrates full-request improvement. Independent local
replay remains a follow-up, not a completed measurement.

A local timing attempt on 2026-10-19 produced no benchmark result and was stopped
when the model was found offloaded by iCloud. No confirmation claim derives from
that attempt. Experiment 8 remains design-only and is outside execution scope.

## Explicit hardware boundary, user clarification 2026-10-19

Arno's dataset is from an Apple M4 Pro with 24 GB unified memory; the
local dataset is from an Apple M2 Pro with 16 GB unified memory. Keep these as
separate hardware cohorts. Do not pool their latency, throughput or memory,
interpret a cross-machine difference as an optimization gain, or treat M4
shader rankings as M2 measurements. M2 phase profiling is complete; its
operation-level GPU attribution remains pending. Completion of the combined
review does not complete that local measurement.

## Local CLI feasibility-pilot outcome, 2026-10-19

Following the user's Xcode setup, Finder downloads, CLI direction and expanded
pilot-storage authorization, clean confirmation completed 40 measured M2 runs
with original prompt/output identity. A final warmed-worker Instruments pilot
was saved before chat interruption and validated on resume. Its metric is
summed target shader Sample Duration, with union coverage and overlap disclosed;
it supports a bounded mixed-workload M2 matrix ranking only. The pilot's three
worker outputs match clean M2, but its third request exceeds the recording
window. It does not meet the original separate-phase GPU attribution gate.
Retain that gate and KV attribution as pending. Publish the completed pilot and
clean confirmation without silently declaring original local replay complete.

## Separate M2 CLI diagnostic contract, 2026-10-20

The user authorized completion of the remaining M2 phase and KV measurements
and approved three short project-local recordings (expected 400-600 MB,
1 GB additional retained ceiling, 2 GB observed working-output emergency stop;
Xcode cache growth remains unknown). This supersedes the pilot-only execution
boundary for these three recordings, without changing the clean benchmark.

Accept equivalent new CLI evidence in place of unavailable original .gputrace
replay when: each phase has an explicit synchronized boundary; work completes
before recording stops; worker/source/trace provenance is retained; output token
hashes match the clean M2 reference; target-only Sample Duration is reported
with its denominator and overlap union; and Mach-clock conversion places the
entire measured window inside the recorded interval. Do not describe these
sampled durations as exact dispatch time or clean request fractions.

Prefill includes all 2,047 prompt tokens and first-output sampling, stopping
before the generator's first decode lookahead model call. Decode excludes
prefill, primes four yielded outputs and one evaluated lookahead, then records
32 new single-token model calls. Neither diagnostic changes the generator's
arithmetic. Both finish 128 outputs and must match the unchanged generator.

KV isolation uses the actual pinned KVCache.update_and_fetch implementation,
16 layers, 8 KV heads, head dimension 64, fp16 inputs, and evaluated synthetic
keys/values. Cover initial allocation with 2,046 input tokens, ordinary append
at offset 2,051/capacity 2,304, and growth at offset/capacity 2,048, five trials
each. Setup and numerical checks stay outside the recorded windows. Report
synchronized update/evaluation wall time and sampled GPU metrics separately;
these isolated costs are not estimates of the cache share in normal inference.

The diagnostic can close with validated per-phase operation rankings,
source-attributed isolated KV costs, and an explicit kernel decision. No kernel
implementation or optimization speedup is required or implied. Retain original
unreplayed captures and disclose remaining metric/generalization limits.

## Acceptance result, 2026-10-20

The corrected separate-phase recordings pass synchronized boundaries, full
GPU-execution containment, output identity and provenance checks. Isolated KV
allocation/append/growth numerical checks and GPU Active interval attribution
pass. Ordinary append has execution intervals but no shader samples; preserve
that distinction. The diagnostic closes under this contract with no custom
kernel selected and no measured speedup. Exact dispatch costs, in-model KV
fractions, additional prompt lengths and independent old-capture replay are
limitations/future replications, not silently completed measurements.

# Experiment 7 completion plan

Date: 2026-10-19
Specification: `experiments/experiment-7-runtime-profiling-spec.md`.
Original protocol: `research/experiment-7-runtime-profiling.md`.

## Ordered work

- [x] 1. Inspect repository history, original protocol, saved phase profile, trace
  metadata, installed tools, and the existing Obsidian project notes.
- [x] 2. Write the specification and this plan before additional measurements.
- [x] 3. Validate the original evidence and local trace sizes; record the audit.
- Superseded 4: environment verified; local confirmation stopped without data
  at the iCloud placeholder. User-directed branch review replaces this run.
- Superseded 5: review the existing M4 replay transcript under the amended
  specification. No new local replay or absolute GPU duration is claimed.
- [x] 6. Record measured results, limitations, and the exact completion state in
  repository documents and Obsidian; preserve a resume task if tooling blocks 5.
- [x] 7. Review intended changes, run the staged-content gate, commit and push,
  then verify the remote commit and Obsidian read-back.

Steps 3 and 4 can proceed while the tooling prerequisite for step 5 is pending.
Step 6 records a blocker if step 5 cannot proceed; that does not mark step 5 or
experiment 7 complete. No new model, dependency, Xcode, or trace download is part
of this plan. Preserve pre-existing working-tree edits.

## Tooling prerequisite

As checked on 2026-10-19, `xcode-select -p` selects CommandLineTools, full Xcode
is absent from standard system/user application locations, and `xctrace` is
unavailable. Existing captures are ready; full Metal Debugger access is required
before operation-level completion. A different installed Xcode location can be
used without changing global developer settings. Installation requires an agreed
storage budget and location under the user's storage rules.

## Execution record

- 2026-10-19: Original phase timing and two local traces found. No source-code
  changes are planned. Specifications separate phase completion from the pending
  operation-level measurement so the final status remains evidence-backed.

## Plan amendment: use Arno's completed M4 evidence

User-directed scope change, 2026-10-19: inspect and use the existing
`ArnoFeatures` branch before repeating work. Pinned source commit:
`1de18115d4f5323c07e4cac850feea1065ef3dee`, author GitHub `axrno`.

The branch is one commit ahead of the original main and contains a pinned M4
phase profile, manually transcribed Xcode shader-cost ranking, two capture
metadata records, and an Experiment 8 design review. Experiment 8 has no runs.

The local confirmation was stopped before producing a timing result because
its existing model weight file is an iCloud `dataless` placeholder. No new model
was downloaded and no completed result will be claimed. Its launch provenance
will be preserved as a partial artifact. A future local rerun requires Finder
Keep Downloaded for the existing model and environment, plus fixed prompt date.

Replace the remaining execution sequence with:
- [x] A. Fetch the pinned branch into existing Git metadata; inspect only the
  small tracked evidence and source changes, without another checkout.
- [x] B. Validate original M2 and both M4 timing files, recompute summaries,
  check prompt/output hashes, audit capture metadata and shader-cost totals.
- [x] C. Incorporate Arno's seven new evidence/research files with source credit;
  adopt and verify the minimal prompt-date fix. Do not import hardware-specific
  project defaults or claim M4 replay percentages as M2/end-to-end timings.
- [x] D. Update repository and Obsidian with the scoped diagnostic conclusion:
  phase crossover repeats across M2/M4; M4 replay reports quantized matrix work
  as the largest shader cost; no custom-kernel win is established. M2 GPU replay
  and independent M4 replay remain unperformed local replications.
- [x] E. Run focused date-fix checks and the staged-content gate, commit and
  push intended files, verify remote state and Obsidian read-back.

This amendment replaces the local rerun and local Xcode replay dependency for
repository evidence consolidation. It does not claim independent reproduction
of Arno's GPU observations. Treat his transcript as reported evidence with an
explicit source, tool, denominator, hardware, and missing raw replay data.

## Validation and documentation checkpoint

- Original M2, pinned M4 and diagnostic M4 profiles passed saved-data checks:
  120 measured requests and 12 warmups; every summary was recomputed.
- Four pinned M4 prompt hashes match M2; the 510-token output differs.
- Both local M2 trace logical sizes match metadata (2,143,001,155 bytes total).
- All seven imported evidence/research files match the pinned source bytes.
- Three focused date-regression tests pass; all benchmark Python files parse
  and the profile CLI exposes the pinned date and override.
- Obsidian experiment, project, decisions, task, daily, board, evidence/research
  maps, index and operation log were updated and read back through the CLI.
  No unresolved links were reported. Publication remains the final checkpoint.

## Publication checkpoint: passed

The substantive update was committed and pushed as
`260062d79710e45bc4f956048a2440143f991cc6`; remote `main` was checked and
matched that commit. The staged-content gate reported no findings across the
19 intended files. Arno's source commit is credited in the review and commit.

Obsidian CLI read-back confirmed the reviewed results, task, project, decision
register, board, daily note, evidence/research maps and index; no unresolved
links were found. The publication task is complete. This final plan-status
update follows the already-verified substantive publication.

No local M2 GPU replay, independent M4 replay, new local timing result, kernel
speedup or Experiment 8 execution is claimed. Those boundaries remain explicit
in the specification amendment and review.

## Hardware clarification follow-up, 2026-10-19

- [x] Record Arno’s M4 Pro / 24 GB and local M2 Pro / 16 GB as separate
  cohorts in the review, project context, README, specification, structured
  audit and Obsidian. M2 GPU operation attribution remains pending.

## Resume M2 measurement and replay, 2026-10-19

User confirmed full Xcode installation and Finder download. Metadata checks
confirm the local model is no longer dataless, Xcode is selected, and package
versions remain MLX 0.32.2 / MLX-LM 0.31.3. Xcode license acceptance and first
launch setup still require the user; no license is accepted by the agent.

- [x] Verify model availability, runtime versions and selected Xcode.
- [x] Run a new clean pinned-date M2 phase profile; preserve exact model/source
  hashes and OS/power provenance, then validate against original M2 inputs.
- [ ] After user finishes Xcode setup, replay the two existing M2 captures,
  retain sanitized GPU attribution and denominator details, and apply the gate.
- [ ] Record the M2-specific result in the repository and Obsidian, run the
  staged-content gate, publish and verify GitHub.

Reuse the existing traces and environment. Only small JSON/text artifacts are
planned; no new model, trace, runtime or alternate checkout is created. The OS
has changed since the original measurements; report confirmation timings as a
new environment observation rather than an optimization-arm speedup.

### M2 resume prerequisite checkpoint

The model files are verified local, but the existing `.venv` still contains
cloud-only `mlx_lm` and `transformers` source files. The new timing attempt was
stopped before its first result; its provenance is an ignored partial artifact.
The user has been asked to Keep Downloaded for the existing `.venv` in Finder.
No environment was installed or relocated.

Xcode is installed at `/Applications/Xcode.app` and selected, but developer
tools report an unaccepted license. User first-launch/license setup is pending.
Computer Use access to Xcode was not approved; no alternative UI-control route
was attempted. Replay remains pending authorized app access or a user-provided
export after setup.

Apple documents the newer shader-cost graph as M3-or-later only. Use the M2's
supported performance/counter/pipeline profiling views instead and record the
actual metric/denominator; do not manufacture an M4-equivalent cost percentage.
Source checked 2026-10-19:
https://developer.apple.com/documentation/xcode/optimizing-gpu-performance

### CLI-only profiling amendment: proposed pilot

User direction: use CLI and allow approximately 20 minutes for the existing
`.venv` iCloud download. Xcode 27.1 (27A9275) now passes first-launch checks.
`xctrace` lists Metal System Trace and Metal GPU Counters, but an in-memory
export probe rejects the saved Metal capture with exit code 11:
`Cannot import files of type 'com.apple.gputools.gputrace'`.
No supported `metal-replay` or `metal-capture` xcrun command was found.

- [x] Verify Xcode first-launch setup and CLI profiling capabilities.
- [x] Wait for the user-managed `.venv` download; check metadata for remaining
  dataless files before starting inference.
- [x] Complete the clean, pinned-date M2 phase confirmation first.
- [x] Subject to a separate storage allowance, run one five-second Instruments
  pilot using Metal System Trace plus Metal GPU Counters and the existing
  runner at target 2,048 tokens, one warmup and one measured repetition.
  Preserve it as instrumented diagnostic data, not clean benchmark timing.
- [x] Export the pilot table of contents to determine whether measured GPU
  durations and useful operation/pipeline labels are available on this M2.
  Counts or CPU API time do not satisfy the operation-time acceptance gate.
- [ ] If the pilot exposes usable timing, plan only the minimum additional
  captures needed; otherwise report the exact CLI limitation and keep local
  GPU attribution pending. Do not replace measurements with M4 shader shares.

Proposed output location: `benchmarks/results/experiment7-m2pro-cli-pilot-*.trace`
and adjacent small export/provenance files. Proposed retained recording budget:
100 MB, with an output-size monitor to stop the pilot at the budget. Peak
transient use and Xcode-managed cache growth are not established by the help
output; no fixed total-disk bound is claimed. Raw recordings remain ignored.
No recording has been started by this capability check. Existing GPU captures
are reused only if a supported CLI reader becomes available.

### Pilot storage approval

On 2026-10-19 the user explicitly approved the one five-second CLI recording
pilot with at most 100 MB of retained recording output. Xcode temporary/cache
growth was disclosed as unknown. This approval applies to this project-local
pilot only, not new large captures or installations. Run the clean phase
confirmation separately and first; wait for the existing environment download.

### Pilot window correction within the approved budget

The launch-based pilot exported successfully but contained zero Python-target
GPU rows; it is excluded from operation attribution. Its retained recording is
49,064,685 bytes. Correct the same pilot by attaching to a process which has
already loaded the model and warmed the kernels. Use xctrace's supported
tracing-start notification and notifyutil's one-notification wait to trigger
work only after tracing starts. Keep combined retained recordings <=100 MB;
discard only a newly generated retry if its final size would exceed the budget.
No existing user artifact is removed. The five-second instrumented window is
a feasibility diagnostic, not a full phase-specific profiling result.

### Final controlled CLI lifecycle retry

The warmed attach attempt triggered MLX work successfully (three stable
128-token outputs), but xctrace did not finalize after its five-second limit
and was stopped at the 120-second command timeout. No GPU attribution is
claimed from that incomplete 40 KB recording. A final controlled retry will
keep the warmed target and Metal resources alive until Instruments finishes,
rather than allowing Python to exit during finalization. This tests the target
lifecycle as the possible cause; it is not a confirmed diagnosis.

The user subsequently authorized exceeding 100 MB if needed. Keep this retry
to five seconds, stop if combined project recording contents exceed 250 MB,
and report actual retained/transient size. Do not start a larger profiling
campaign. If finalization still fails, preserve the exact CLI limitation and
publish the validated clean phase result with GPU attribution still pending.

### Clean M2 confirmation checkpoint: passed

All 40 measurements and four warmups completed. All four input and output
hashes match the original M2 run; summary recomputation passed. The same phase
crossover repeats. The new OS and battery provenance are explicit, so timing
shifts are not treated as optimization-arm gains. `.venv` is fully local.

### Correct the temporary-space guard before completing the pilot

The held-target attempt reached recording completion and saving, but its
conservative 250 MB combined temporary-size guard interrupted saving; export
then reported malformed data. Exclude that artifact. The user has allowed
exceeding 100 MB if needed. Repeat only the same five-second window while
allowing finalization to finish naturally (120-second command timeout and a
1 GB emergency output threshold). Based on observed files, expect roughly
170 MB combined retained recordings and a larger temporary working footprint.
This corrects an agent-imposed premature-stop condition, not an unsupported
claim that the recorded GPU data was valid. Validate the resulting export.

### Resume checkpoint: saved pilot validated and publication prepared

- [x] Resume the final saved recording; no fresh capture needed.
- [x] Validate the export and isolate Python target rows using resolved XML IDs.
- [x] Save reproducible, sanitized Sample Duration aggregation with interval unions.
- [x] Independently recheck 40 clean runs, four warmups, summaries, hashes and timing identities.
- [x] Correct the draft power-source error: recorded AC power, not battery.
- [ ] Synchronize Obsidian, pass staged-content gate, commit/push and verify remote.

Final recording: 137,245,510 logical bytes; combined retained pilot recordings:
247,936,324 bytes; observed combined peak: 383,709,630 bytes. Xcode cache and
unobserved temporary growth remain unknown. Three complete worker outputs
match clean M2 output, but the third request extends beyond the recording.
Quantized matrix/dequantization is 89.67% of summed target shader Sample
Duration; union coverage and overlap are disclosed in the analysis. Do not
interpret this as exact kernel time or an end-to-end fraction.

The authorized pilot is complete. Original separate-phase M2 GPU attribution
remains pending, as does independent KV-update attribution. The saved capture
reader rejects `.gputrace`; a larger profiling campaign is outside this pilot.
Keep the broader M2 task in progress with this precise remaining scope.

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

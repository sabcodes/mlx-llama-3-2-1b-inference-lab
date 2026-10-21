# Experiment 8 execution plan: packed quantized projections on M2

Date: 2026-10-20
Status: proposed; design published separately from execution. No runs/results.
Specification: [experiment-8-packed-projections-spec.md](experiment-8-packed-projections-spec.md).

## Intent

Test one concrete consequence of the completed local Experiment 7 diagnosis:
pack same-input projections using the existing quantized matrix path. Keep the
M2/1B deployment and weight values fixed. This is a local proposal independent
of the historical M4 speculative-decoding reference design.

## Design/publication checklist (this request)

- [x] Review local phase/KV results, pinned Llama and quantized-layer source.
- [x] Define the hypothesis, C/Q/G/QG arms, invariants and correctness gates.
- [x] Predeclare a bounded screen, one-finalist selection and full confirmation.
- [x] Separate natural-EOS and forced-output workload identities and new controls.
- [x] Specify request/packing memory, startup, tail and rejection rules.
- [x] Save the proposal in Obsidian and update repository navigation/context.
- [x] Validate links and scope; pass staged-content gate; publish and verify GitHub.

## Ordered execution checklist (future work; not started)

- [ ] 0. Recheck source/model hashes, versions, exact source shapes, local iCloud
  availability, storage budget and stable power conditions. Record a run manifest.
- [ ] 1. Add the minimum experiment-local row-packing adapter. Materialize once,
  preserve all quantization fields and output splits, release old references,
  leave embedding/output head and downstream operations unchanged.
- [ ] 2. Prove byte identity and numerical behavior on all layers and declared
  boundary shapes. Measure logical parameter bytes before/after and packing peak.
- [ ] 3. Pass exact whole-model token/EOS/finish and history/cache correctness
  checks for each arm. Stop each failing arm; retain failure evidence.
- [ ] 4. Screen surviving C/Q/G/QG arms on the four summarization lengths,
  forced 128 tokens, one warmup and five measurements. Freeze at most one finalist
  using the specification's deterministic rule; stop if none survives.
- [ ] 5. Run a new C/finalist diverse comparison: 20 fixtures, 10 measurements
  each, normal EOS/cap 128, fresh cache and identical prompt hashes.
- [ ] 6. Independently repeat with reversed arm order, same gates and fresh
  processes. Do not aggregate away a failed capture or change gates post hoc.
- [ ] 7. Audit completion, raw metrics, outputs, memory and startup. Classify
  reject/inconclusive/select; record limitations and the exact scope of any win.
- [ ] 8. Update analysis/context/Obsidian and publish intended sanitized artifacts.
  A deployment/default change is separate from selecting a benchmark candidate.

## Measurement inventory

| Stage | Cases and repetitions | Purpose |
|---|---|---|
| Source/packing checks | All 16 layers and all specified boundary shapes | Confirm the actual mechanism and unchanged quantized values |
| Correctness | 20 natural-EOS fixtures, 4 forced-128 cases, 1 forced-256 case, all existing multi-turn scenarios per surviving arm | Exclude token/cache regressions before selection |
| Bounded screen | Up to 4 arms x 4 lengths x 5 measured requests; 16 warmups | Reject cheaply and freeze one finalist |
| Full capture | 2 arms x 20 fixtures x 10 measured requests; 40 warmups | Test diverse full-request benefit |
| Confirmation | Repeat full capture with reversed arm order | Require reproducibility rather than a single favorable run |

Layer microbenchmarks are diagnostic; the promotion gate is >=10% full-request
median improvement at both short target lengths, with all correctness, tail,
per-fixture, long-prompt and memory constraints passing independently twice.
Exact acceptance rules and data semantics live in the specification.

## Outputs and storage

- Implementation, when separately requested: small source under `benchmarks/`.
- Future results: unique `experiment8-packed-<arm>-<stage>-<id>` files under
  `benchmarks/results/`; partials remain ignored. Record workload IDs and seeds.
- Interpretive report after runs, not a fabricated result stub now.
- Existing checkpoint/environment reused; no model conversion files or installs.
- No GPU captures in the primary plan. Any later diagnostic capture needs its
  own agreed storage allowance. Proposed timing-only retained output <=10 MB.
- Current design publication changes only small Markdown documents and private
  vault notes. No benchmark, installation or packed tensor allocation is run.

## Predeclared stop outcomes

- Unsupported packing, changed quantization fields, wrong ordering or tensor
  duplication: reject implementation before timing.
- Numeric/token/cache/required-format mismatch: reject that arm.
- No clean full-request screen improvement: stop without a full benchmark.
- Memory, tail, per-fixture or long-prompt gate failure: do not select a default.
- Conflicting repeats or ambiguous evidence: inconclusive; register a new
  confirmation question instead of rerunning until a preferred outcome appears.

## Rationale for this scope

It targets the measured matrix family with one specific existing-API change.
Alternative quantization formats were already screened in Experiment 4; a
second full-size draft already lost in Experiment 6. Custom Metal code and
cross-runtime/model changes would add variables before this simpler hypothesis
is tested. The proposal makes no claim that any alternative is permanently
unhelpful and does not inherit another machine's experiment sequence.

## Design publication checkpoint: passed

Published the specification, plan and repository navigation/context changes at
`e78e23bd9e6d70decb7594ee5809f9d4b7ad19e3`; remote main was verified equal.
All local document links, source-derived dimensions and the approximately
342 MiB naive packing estimate were checked. The staged-content security gate
reported zero findings across the four intended Markdown files. Obsidian
experiment/task/project/decision/evidence/board/index/daily/operation notes were
updated and read back with no unresolved links. Design is Done; execution stays
planned. All unchecked execution items above are intentional future work.
No runtime implementation, benchmark, installation or model allocation ran.

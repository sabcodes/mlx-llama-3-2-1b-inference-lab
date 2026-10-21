# Experiment 8 execution plan: packed quantized projections on M2

Date: 2026-10-20
Status: completed 2026-10-20; tested arms rejected at predeclared gates.
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

## Ordered execution checklist (completion record)

- [x] 0. Rechecked local model/runtime/source/device, storage, power and capacity;
  final r3 captures retain immutable preregistration/source snapshots.
- [x] 1. Added experiment-local row packing; no site-packages or checkpoint edits.
- [x] 2. All 16 layers/families pass exact component bytes and numerical grids;
  parameter bytes remain unchanged.
- [x] 3. Evaluated whole-model/history/cache gates. G passes 65 cases; Q/QG
  stop at their first exact-token mismatch, with rejected evidence preserved.
- [x] 4. Screened the surviving C/G arms with four forced-128 fixtures and five
  measurements each. Both short benefit and request-memory gates reject G.
- [x] 5. Conditional full diverse comparison skipped: no eligible finalist.
- [x] 6. Conditional reverse-order confirmation skipped: no eligible finalist.
- [x] 7. Audited raw metrics, source, order, capacity and outcome independently.
  All tested arms rejected; no general speed/quality claim or default change.
- [ ] 8. Final publication/readback checkpoint; analysis, context and private
  knowledge records updated. Deployment is a separate change.

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

## Execution preregistration amendment (2026-10-20, before implementation)

The user now authorizes implementation and execution in this project. Follow
the existing ordered checklist, with adversarial method/source review before
timing and a separate result audit after execution. First update this plan,
then the specification, then implement the experiment-local adapter and runner.
Reuse the local checkpoint/environment. Retained timing/check JSON and logs
are bounded to 10 MB; no new model/runtime, exports, GPU traces, or external
location. The model is downloaded; free space is approximately 32 GiB.

Add source-derived dispatch-boundary tests (9/10/11, 12/13, 17/18/19) to
the existing shapes. Validate Q and G families independently; QG must pass
both and its own whole-model gate. Stop on the first hard failure per arm,
preserve its exact failure, and skip later gates/timing for rejected arms.
Do not reinterpret a control quality defect as an optimization regression:
record control defects separately; candidate must preserve exact outputs and
introduce no new required-fact/format defect. A known control defect is a
limitation on selecting a useful deployment, not evidence of candidate quality.

Record AC/battery/low-power and swap state at process boundaries and check
swap/RSS between cases. Hold power mode fixed; stop timing on a changed state.
Synchronized request end includes generation-stream lookahead completion.
Cache inspection/materialization and output checks occur after that boundary
and remain identical for every arm. Do not retain response logprob arrays.
Screen order seed is 20261020; stage-2 order remains C/finalist then finalist/C.
All source files and the frozen plan/spec are hashed in each result manifest.
Unknown counters or unavailable measurements are explicit, never assumed zero.

Unconventional ideas are evaluated in a cited research feasibility matrix;
new mechanisms/weights/runtimes require a separately registered experiment.
This experiment exhausts its declared C/Q/G/QG arms, not all possible inference
optimizations. Any later scope extension requires new predeclared gates.

Final preimplementation review details: input dtype follows the checkpoint
embedding activation dtype; bounded random inputs use MLX seed 20261020 and
uniform [-1,1]. Capture layer-normalized attention and MLP activations from
the fixed 2,048-target summarization fixture; keep at most 65 rows per layer
and tile those rows for larger diagnostic shapes. All shapes are sequential.
Use fixed arm rank Q,G,QG as final selection tie-break; never promote a second
finalist after confirmation failure. Record first differing token and preceding
control/candidate top-two logprob margins if available. Before each timing
process use the same 10-second idle interval and record power/thermal status;
second full pair reverses fixture order. Shared control quality defects prevent
an absolute quality-pass/selection claim, while equivalence remains reportable.

## Execution outcome, 2026-10-20

All declared arms are resolved. Q/QG pass numerical tolerance and exact packed
bytes but fail the exact-token gate. G passes all 65 generation/history/cache
cases, then fails the >=5% short-request screen benefit and <=105% request
peak-memory gates. No finalist remains. Conditional full comparison and
reverse-order confirmation are intentionally skipped, not unfinished work.
No gate was loosened and no selected runtime setting changes.

The [final analysis](../benchmarks/results/experiment8-packed-projections-analysis.md)
records raw results, excluded swap/setup attempts, the first-token diagnostic,
validator limitations and independent reviews. Immutable preregistration and
source snapshots accompany r3 captures; this live document now includes outcome.

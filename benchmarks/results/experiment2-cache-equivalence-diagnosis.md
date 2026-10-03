# Experiment 2 cache-equivalence diagnosis

Date: 2026-09-25

## Outcome

The long-context output mismatch is caused by numerical sensitivity to different prompt-evaluation batch/chunk shapes, not by a wrong token prefix, an off-by-one cache boundary, or stale LRU cache state.

Under the experiment's predeclared strict token-equivalence requirement, the KV-prefix-reuse treatment fails its correctness guardrail. The partial treatment capture must not be used as a completed timing comparison. Telemetry remains deferred because it would measure a treatment that has not passed correctness validation.

## Minimal deterministic reproduction

The `repair-workshop` fixture reproduces the mismatch at turn 2 after a 2,056-token first-turn prompt and 73 generated tokens. Its next fully rendered prompt contains 2,148 tokens. The prefix matcher reports 2,129 reusable tokens and a 19-token suffix.

The saved control and treatment:

- use identical turn-2 full input token IDs;
- use identical first-turn generated token IDs;
- have cache logical length equal to the stored cache-key length;
- first differ at generated token index 53 (zero-based);
- produce 84 control tokens and 82 reuse tokens for that turn.

The diagnostic rebuilds the cache without using `LRUPromptCache`. It evaluates the same known token sequence with the exact segment schedule used by the earlier cached generation, then supplies the 19-token turn-2 suffix. This independently reconstructed path matches the saved reuse output exactly. A fresh full-prompt evaluation matches the saved control output exactly.

| Diagnostic comparison | Result |
| --- | --- |
| Fresh full prompt vs. saved control | Exact match |
| Independently segmented prefix vs. saved reuse | Exact match |
| Fresh full prompt vs. independently segmented prefix | Different |

This isolates evaluation shape/order as the changing variable and rules out the LRU lookup as the source of the divergence.

At the first differing generation position, the fresh path chose token ID `4819`; its logged candidate log-probabilities were `-1.0` for token `4819` and `-1.015625` for token `4754`, a margin of `0.015625`. The segmented path logged `-1.015625` for both candidates, and greedy decoding selected token `4754`. Once one greedy token differs, all later model state and output may diverge.

These values establish a near-tie that changed under the alternative batching schedule. They do not establish a general error bound for MLX-LM or this model.

## Implication for the experiment

Mathematically, both paths represent the same token context. Computationally, they reach it through different matrix shapes and operation groupings: the control prefills the current full prompt in large chunks, while the treatment retains work produced by the earlier turn and then evaluates only the new suffix. Finite-precision GPU computation need not produce bit-identical logits across those schedules. Greedy decoding makes a small near-tie visible as a different token.

The approved experiment required exact generated-token equality. Therefore:

1. Keep the completed full-recompute capture as the Experiment 2 control baseline.
2. Keep the stopped reuse artifact labeled partial and exclude it from performance conclusions.
3. Do not collect the telemetry pair yet.
4. Treat exact token equality as incompatible with this observed long-context cache-reuse path unless a later implementation makes the evaluation schedule identical or otherwise proves bitwise determinism.
5. If the experiment is later redesigned to test practical chat quality, define and approve a semantic/correctness tolerance before looking at performance results; that would be a new protocol decision, not a silent relaxation of this experiment.

## Artifacts

- Completed control: `experiment2-control-timing-20260925T165100Z.json`
- Partial reuse attempt: `experiment2-reuse-timing-20260925T165355Z.json`
- Minimal control: `experiment2-debug-repair-control-20260925T165652Z.json`
- Minimal reuse: `experiment2-debug-repair-reuse-20260925T165702Z.json`
- Isolating diagnostic: `experiment2-debug-repair-cache-equivalence-20260925.json`
- Diagnostic runner: `benchmarks/diagnose_experiment2_cache_equivalence.py`

# Apple Silicon inference engineering for Llama 3.2 1B

This repository applies a repeatable inference-engineering process to
`mlx-community/Llama-3.2-1B-Instruct-4bit` on an Apple M2 Pro with 16 GB unified
memory. It changes and measures the MLX/MLX-LM inference path; it does not train
or fine-tune the model.

The project currently keeps the published affine 4-bit/group-64 weights, uses
a 2,048-token prompt-prefill step, and selects exact-prefix KV reuse for
sequential multi-turn chat where appropriate. These are benchmark-backed
runtime decisions; the repository does not yet contain a production chat
server that deploys them. Other tested changes were rejected or retained only
as diagnostic evidence.

Start with [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the current decisions,
known limitations, experiment map, and next work.

## Repository map

- `benchmarks/`: repeatable runners, deterministic fixtures, raw JSON results,
  and experiment analyses.
- `experiments/`: predeclared experiment plans and decision rules.
- `research/`: primary-source research and design notes.
- `.agents/skills/`: reusable benchmark and prefill-tuning procedures.

The baseline protocol is documented in
`.agents/skills/mlx-inference-benchmark/references/benchmark-protocol.md`.

## Current experiment decisions

| Experiment | Decision |
|---|---|
| Baseline | Preserve the diverse five-task × four-length reference capture. |
| 1: prefill step size | Keep 2,048; 512 is a memory-oriented alternative. |
| 2: cross-turn KV reuse | Adopt for exact-prefix sequential chat with careful cache lifecycle. |
| 3: explicit compiled decode | Reject the tested dynamic-cache compilation seam. |
| 4: weight quantization | Keep published affine 4-bit/group-64. |
| 5: KV-cache quantization | Reject all tested arms under the approved gates. |
| 6: speculative decoding | Reject the full-1B Mixed-3/6 draft pairing. |
| 7: runtime profiling | Complete: separate M2 phase rankings and isolated KV costs validated. Quantized matrix work dominates sampled prefill/decode; no custom-kernel win established. |
| 8: packed quantized projections | Reject tested arms: Q/QG fail exact tokens; G passes correctness but fails the bounded speed/memory screen. |

## Local setup

The repository expects, but intentionally does not contain:

- `.venv/`
- `models/Llama-3.2-1B-Instruct-4bit/`

From the repository root, the stable baseline is run with:

```bash
./.venv/bin/python benchmarks/run_baseline.py \
  --output benchmarks/results/<unique-run-id>.json
```

Never overwrite a prior result. Keep primary timing runs separate from
`powermetrics` or Metal-capture diagnostics.

## Data and privacy policy

Model artifacts, virtual environments, incomplete captures, and raw Metal GPU
traces remain local. Raw `.gputrace` directories are large and may contain
machine-local metadata. Committed telemetry logs have process-name and PID
tables removed.

Experiment 7's [review and provenance](benchmarks/results/experiment7-review-20261019.md)
incorporate [Arno's M4 Pro / 24 GB evidence](benchmarks/results/experiment7-m4pro-runtime-profile-analysis.md)
and a fixed chat-template date for repeatable prompts. GPU percentages are
reported replay shader-cost shares, not request-time savings. The
[historical M4 speculative-decoding design](research/experiment-8-and-followups-review.md)
is reference material, separate from the local packed-projection proposal.
Local Experiment 8 is complete at its declared elimination gates; see the
[results](benchmarks/results/experiment8-packed-projections-analysis.md).

Arno's M4 Pro / 24 GB results and the local M2 Pro / 16 GB results remain
separate hardware datasets.

The [completed M2 phase/KV diagnostic](benchmarks/results/experiment7-m2pro-phase-and-kv-analysis-20261020.md)
adds separately bounded prefill and decode recordings plus isolated KV update
measurements. Quantized matrix/dequantization accounts for 91.53% of prefill
and 88.49% of decode summed shader Sample Duration. These are instrumented
sample shares, not clean request fractions or measured speedups. The original
`.gputrace` files remain unreplayed; new validated CLI evidence closes the
diagnostic under the documented October 20 contract.

Experiment 8 followed its [specification](experiments/experiment-8-packed-projections-spec.md)
and [ordered plan](experiments/experiment-8-packed-projections-plan.md). QKV
packing and combined packing fail exact output preservation. Gate/up packing
passes 65 generation/history cases but achieves no >=5% short-request screen
benefit and raises request peak memory about 10–12% at longer lengths. No
finalist remains, so full confirmation is intentionally skipped. The
[analysis and limitations](benchmarks/results/experiment8-packed-projections-analysis.md)
include the exposed first-token tie, control-validator limitations, independent
audits and the next research questions. Selected runtime settings are unchanged.

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

Experiment 7 was repeated on an M4 Pro with 24 GB. Its Xcode Metal replay
identifies the dominant GPU shader families.

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
| 7: runtime profiling | M4 phase split and GPU shader ranking measured; a custom kernel still needs a faster tested candidate. |

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
traces remain local. Raw `.gputrace` and Instruments `.trace` directories are
large and may contain machine-local metadata. Committed telemetry logs have
process-name and PID tables removed.

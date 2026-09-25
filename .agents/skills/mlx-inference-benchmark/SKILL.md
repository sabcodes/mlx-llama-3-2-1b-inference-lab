---
name: mlx-inference-benchmark
description: Run and compare the project's repeatable MLX LM benchmarks for Llama 3.2 1B while keeping workload and measurement conditions consistent. Use when capturing a baseline, testing an inference optimization, or collecting Apple Silicon GPU and thermal telemetry.
---

# MLX inference benchmark

Use this skill to measure changes to the inference path against a stable, repeatable workload. Read [the benchmark protocol](references/benchmark-protocol.md) before running or changing the suite. Read [Apple Silicon telemetry](references/apple-silicon-telemetry.md) when GPU or thermal data is requested.

Keep the existing model weights, project `.venv`, prompt fixtures, sampling settings, and default lengths fixed for comparisons. Do not download or alter the model, update dependencies, or add a privileged telemetry collector without the user's direction. Treat a deliberate workload change as a new benchmark version, not as a directly comparable run.

Run from the project root with the project-local environment:

```sh
./.venv/bin/python benchmarks/run_baseline.py \
  --output benchmarks/results/<unique-run-id>.json
```

Choose a new descriptive run ID for every capture; never overwrite a prior result. The default suite performs one cold first inference, one warmup, and 10 measured runs for each prompt fixture. Check that a completed default result contains four input-length cases, five prompt fixtures per case, 10 measured runs per fixture (200 total), summaries, and `completed_at_utc`. Treat a file without those completion markers as partial.

For an optimization comparison, record what changed and its revision alongside the run ID, run the unchanged suite before and after, and report the per-prompt and pooled-by-length medians and p95s. Keep ordinary timing captures separate from privileged/system telemetry captures because instrumentation can affect performance. Report unsupported measurements as unmeasured; do not substitute estimates or Activity Monitor screenshots for numeric GPU utilization.

When telemetry is requested, show the user the collector command from the telemetry reference and wait for them to start it in their own macOS Terminal. Do not launch a `sudo` prompt in a Codex terminal or ask the user to enter an administrator password into a Codex-controlled session. After the user confirms sampling has begun, launch the benchmark and associate its JSON with the matching telemetry log.

---
name: mlx-prefill-step-size
description: Set up and run a controlled MLX-LM prompt-prefill step-size sweep on Apple silicon, then select a setting from latency, throughput, and memory evidence. Use when tuning prompt prefill chunking for an MLX-LM model or reproducing this project's Experiment 1.
---

# Tune MLX-LM prefill step size on Apple silicon

Use this to find a good prompt-prefill chunk size for a particular Mac, model, MLX-LM version, and workload. There is no universally fastest setting. A result applies only to the tested hardware, software, model, prompt lengths, and measurement conditions.

Read [the procedure and Experiment 1 evidence](references/procedure-and-evidence.md) before setting up or running a sweep. In this repository, use [the repeatable benchmark skill](../mlx-inference-benchmark/SKILL.md) and its protocol for the canonical Llama 3.2 1B workload. Preserve its fixtures and controls; this skill changes only the prefill-step-size setting.

Keep weights, prompt suite, generation settings, dependency versions, and machine conditions fixed. Use a project-local Python virtual environment; never install MLX packages into a global or unrelated environment. Do not download/modify weights, change package versions, run privileged telemetry, or overwrite previous results without user direction.

For a controlled comparison, run the same benchmark once per candidate setting, with a unique result file for each. For this project, candidate settings are 512, 1,024, 2,048, and 4,096 tokens; 2,048 is the installed MLX-LM default, not a guaranteed optimum. Keep telemetry off for primary timing runs. Change no other variable.

After the sweep, compare per-prompt and per-input-length median and p95 TTFT, prefill tokens/second, request time, and memory. Select based on the actual objective and memory budget, not a single pooled speed number. Treat a 10-run p95 as a coarse tail indicator. Rerun or interleave finalists if run order or noisy outliers could affect the decision. Keep raw JSON, record the exact setup and revision, and explain the chosen trade-off.

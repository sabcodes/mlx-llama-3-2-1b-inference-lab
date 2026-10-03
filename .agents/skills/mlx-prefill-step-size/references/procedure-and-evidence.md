# Procedure and Experiment 1 evidence

## What the setting means

`prefill_step_size` is the largest prompt chunk MLX-LM handles at a time while building the model's key/value (KV) cache from input tokens. It does not shorten the prompt, discard earlier text, or limit the context window. The full tokenized prompt remains part of the request; chunking changes how it is processed. The KV cache accumulates the state needed to continue generation. A fresh-cache test measures prompt prefill, not reuse of a prior conversation prefix.

Chunk size trades temporary memory against compute overhead and latency. A smaller step can reduce peak memory but may add work; a larger step can process longer segments but use more temporary memory or perform worse. These are hypotheses to measure, not directional guarantees.

## Setup

1. Confirm the Mac is Apple silicon and record its chip, unified-memory capacity, macOS version, power/charger state, and whether other GPU-heavy work is running. Keep those conditions stable across candidates.
2. Create a virtual environment inside the project/repository and install MLX-LM into it. For reproducing this project's current capture, the tested versions were Python 3.14.4, MLX-LM 0.31.3, and MLX 0.32.2; the corresponding isolated setup is:

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install 'mlx-lm==0.31.3'
   ```

   For a new MLX-LM release or another project, use that project's pinned dependency instructions and check the official compatibility requirements first. Activate that environment (or invoke its Python by path) for all commands. Do not reuse a global environment or one from another project.
3. Record Python, `mlx`, and `mlx-lm` versions, model identifier and weight format, repository revision, and model file identity. Do not upgrade dependencies during the sweep.
4. Verify that the exact installed MLX-LM entry point accepts `--prefill-step-size`, or that the pinned Python generation API accepts `prefill_step_size`. Check the installed version's help/source: command-line flags and Python APIs can evolve. In this project the benchmark script exposes `--prefill-step-size` and writes the selected value into the result metadata.
5. Load the model once per benchmark process. Warm up consistently, avoid concurrent GPU workloads, and give every candidate a new output path.

Use first-party references for current commands and API semantics:

- [MLX-LM repository and installation instructions](https://github.com/ml-explore/mlx-lm)
- [MLX-LM generation implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
- [MLX repository](https://github.com/ml-explore/mlx)
- [Version-scoped install/API research note for this project](../../../../research/mlx-lm-prefill-configuration.md)

Verify the cited interface against the version actually installed; a `main`-branch page may differ from a released version.

## Workload and controls

Use representative prompts and input lengths from the target application, rendered with the model's actual tokenizer and chat template. Include short, medium, and long inputs that resemble production use. Include multiple prompt/task types if the result is meant to generalize beyond one prompt. Record the exact prompt or a stable fixture ID and the actual token count for every case.

For an apples-to-apples sweep, hold constant:

- model files, tokenizer, chat template, and model load behavior;
- exact prompt fixtures, actual input tokens, output-token cap, and sampling parameters;
- concurrency, cache policy, warmups, number of measured repetitions, and metric code;
- software versions, Mac/power conditions, and whether telemetry is enabled.

Change only the prefill step size. Keep prior-prefix/prompt cache reuse disabled for a chunk-size sweep; cache reuse is a separate experiment and requires a multi-turn workload. For this project, reuse the five prompt types at each of 64, 512, 2,048, and 4,096 target input tokens, greedy decoding, a 128-token generation cap, concurrency one, one model load, fresh cache per request, one warmup, and 10 measured repetitions per fixture. Its existing runner performs the same prompt for all repeats and records actual token lengths.

If no benchmark harness exists, build one that logs raw per-request data and environment/settings metadata before collecting results. Do not rely only on aggregate console output. The project harness is `benchmarks/run_baseline.py`.

## Run the project sweep

From the repository root, use the local environment and a unique output path per setting:

```sh
./.venv/bin/python benchmarks/run_baseline.py \
  --prefill-step-size 512 \
  --output benchmarks/results/prefill-step-512-<run-id>.json
```

Repeat with `--prefill-step-size 1024`, `2048`, and `4096`, changing only the value and output name. Run one candidate at a time. Never overwrite or edit the saved reference result. Check each JSON for completion markers, all expected fixtures and repetitions, and a recorded step-size value. Interrupted files are partial and must not be included as completed samples.

For another model/workload, adapt the harness to record exact input and generated token counts, and ensure its step-size parameter is applied to every compared request. Make the candidate range cover values below and above the library's documented/current default, constrained by the workload and memory budget. Do not assume this project's candidate set suits another model or machine.

## Metrics to capture

Capture raw measurements per prompt and summarize at least:

- **Time to first token (TTFT):** user-visible time from request start to first generated token; it includes prefill and runtime overhead.
- **Prefill throughput:** input tokens processed per second, using the same definition across candidates.
- **Decode throughput:** generated tokens per second after first token. This setting primarily targets prefill; decode change is a useful control/sanity check.
- **Request duration:** total generation time; interpret together with the number of generated tokens and EOS/token-cap finish reason.
- **Memory:** MLX active/peak memory and process peak RSS if available. RSS high-water marks may be process-lifetime values, not a per-request delta. Record their semantics.
- **Workload and correctness metadata:** actual prompt tokens, generated tokens, finish reason, output/hash (or other task-appropriate correctness check), exact step size, cache policy, versions, model, hardware, and timestamps.

Report median and a consistent p95 method per fixture and input-length bucket; also report spread/sample counts. The project's runner uses nearest-rank p95. At 10 repeats per prompt, p95 is the maximum, so do not treat it as a precise tail estimate. Its five task types pooled at one input length yield 50 observations and a somewhat more informative, but still limited, tail summary. Never hide a slow fixture inside only a pooled average.

GPU utilization, internal chip temperature, and thermal pressure are distinct measures. The benchmark runner does not produce a validated whole-GPU utilization percentage. `powermetrics` may provide device-level GPU activity/power and categorical thermal pressure, but those are sampled diagnostics, not exact per-request GPU utilization or internal die temperature. Run it only as a separate, optional, user-started privileged capture if requested; see [the project's telemetry instructions](../../mlx-inference-benchmark/references/apple-silicon-telemetry.md). Do not mix instrumented timing runs into the clean primary comparison.

## Selecting a setting

1. Exclude partial or invalid runs and verify each candidate has the same fixture/repetition counts and identical non-step-size settings.
2. Compare per-prompt results first, then per-length summaries. Focus on TTFT and prefill tokens/second at lengths that matter to the intended use; review request duration and decode throughput as secondary behavior.
3. Compare memory against an explicit budget. A modest latency loss may be worthwhile if it materially reduces peak memory; conversely, a fast candidate that risks memory pressure may not be robust.
4. Examine variance, outliers, thermal state, and execution order. If the sweep was sequential in a fixed order or telemetry-instrumented, it is exploratory. Repeat finalists in a different/interleaved order under timing-only conditions before selecting.
5. State the objective and trade-off (for example, lowest median long-prompt TTFT subject to peak memory below a chosen ceiling). Pick the setting that best meets that objective, not a universal winner. If candidates are within observed noise, report a tie and prefer the default or simpler choice.
6. Preserve every raw result and write a concise analysis with test conditions, versions/revision, per-length tables, selection rationale, limitations, and next validation. A setting is a candidate until confirmed on the real interactive workload.

## Experiment 1: this project, this machine

The 2026-09-25 sweep used `mlx-community/Llama-3.2-1B-Instruct-4bit` on this project's M2 Pro / 16 GB MacBook Pro, MLX 0.32.2 and MLX-LM 0.31.3. Values tested were 512, 1,024, 2,048, and 4,096. The suite had four input lengths (64, 512, 2,048, 4,096), five task types per length, one warmup and 10 measured repetitions per fixture, greedy decoding, up to 128 generated tokens, concurrency one, and a fresh cache per request. `powermetrics` was running during these captures, so timings are instrumented and not directly comparable to the earlier timing-only baseline. Candidates ran in ascending order.

Observed result: 2,048 had the best median prefill throughput at 2,048- and 4,096-token prompts and lowest median TTFT at 4,096; short-prompt timing differences among 512, 1,024, and 2,048 were small. The 512 step used less MLX peak memory at long input lengths (about 9% less at 4,096 tokens and 13% less at 2,048 than 2,048), with some throughput trade-off. The 4,096 step was slower across the tested lengths and had a substantial long-prompt latency outlier; cause is unknown. Thermal-pressure samples were all `Nominal`; logs lacked internal die temperature and the per-process GPU-time field was zero. Device-level GPU activity/frequency and power were available but should be interpreted as sampled diagnostics.

The provisional choice for this project's next experiments is 2,048, not a universal optimum. Reconfirm with a timing-only, order-controlled repeat before using it as a production setting. Full tables and raw artifacts are in [Experiment 1 analysis](../../../../benchmarks/results/prefill-step-size-analysis.md).

## What to save with every sweep

Keep immutable raw JSON for every run, plus a human-readable analysis. Record:

- goal, hypothesis, experiment ID, tested candidates, date/time, and repository revision;
- hardware/memory/macOS, Python/MLX/MLX-LM versions, model ID and weight identity;
- exact workload/config, warmup/repetition counts, token counts, and cache policy;
- raw per-run metrics, summaries, output/correctness checks, and optional telemetry association;
- exclusions, deviations, result limitations, selection objective, and chosen trade-off.

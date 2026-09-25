# Benchmark protocol

## Workload

The benchmark targets `mlx-community/Llama-3.2-1B-Instruct-4bit`, stored at `models/Llama-3.2-1B-Instruct-4bit`, and uses the project's `.venv`. The model and environment are gitignored. Use the existing local copies; do not fetch weights or change package versions as part of a measurement run.

The source of truth for fixture construction and task wording is `benchmarks/run_baseline.py` (`DEFAULT_TARGETS`, `PROMPT_TYPES`, `make_fixture`). The default suite contains five single-user chat tasks at each of four prompt lengths:

The accepted expanded reference capture is `benchmarks/results/diverse-baseline-20260925T053031Z.json`. Compare future runs to its per-prompt and pooled-by-length summaries; retain the earlier `baseline-20260925T051846Z.json` as the original single-prompt baseline, not as an equivalent workload.

| Target input tokens | Prompt tasks |
| ---: | --- |
| 64 | Q&A, rewrite, context question, extraction, summarization |
| 512 | same five tasks |
| 2,048 | same five tasks |
| 4,096 | same five tasks |

The tokenizer renders each fixture with the model's chat template. Actual token lengths are recorded; the short fixtures may differ from their target by a few tokens. Every repetition within one fixture uses the exact same prompt. Different fixtures vary both task wording and filler text, so use per-prompt summaries as well as pooled summaries.

Default inference conditions are greedy decoding (`temperature=0`), at most 128 generated tokens, one request at a time, one model load for the process, no prompt cache, and a fresh generation cache per request. The runner captures a cold first inference for the first 64-token Q&A fixture, then one warmup and 10 measured repetitions for every fixture. EOS may end generation before the 128-token cap.

## Run and preserve results

From the repository root, use a new output path for every run:

```sh
./.venv/bin/python benchmarks/run_baseline.py \
  --output benchmarks/results/<model-or-change>-<unique-run-id>.json
```

The script also supports `--warm-runs`, `--max-new-tokens`, and `--targets`; changing any of these changes the experiment conditions. Keep defaults for like-for-like comparisons, or explicitly use identical non-default values for every compared run. Preserve each raw JSON artifact. Record the tested change and Git revision in the run ID or adjacent experiment notes because the current JSON does not identify an optimization patch.

For a prefill-step-size experiment, pass `--prefill-step-size <N>` (default: 2048). Keep the full suite and every other setting unchanged, use a unique output path per value, and run the values sequentially to avoid GPU contention. When collecting `powermetrics`, use the same run ID for its log and the matching JSON; read [Apple Silicon telemetry](apple-silicon-telemetry.md) and wait for the user to start the collector before starting each benchmark. Telemetry-instrumented timings are separate from clean timing-only comparisons.

Before reporting a run as complete, verify that the JSON parses and has four cases, five prompts in each case, 10 measured runs per prompt, one warmup per prompt, summary data, and `completed_at_utc`. A default run therefore has 20 warmups and 200 measured generations. Label interrupted captures as partial and exclude them from comparisons.

## Captured metrics

The JSON stores environment and model metadata, model-load time, prompt messages and actual token counts, each generated output and its hash, and per-run measurements:

- Time to first token, request duration, prefill tokens/second, and decode tokens/second, as reported by MLX LM's streaming generation responses and measured around the request.
- MLX active and peak memory, plus process peak resident set size. Process peak RSS is a process-lifetime high-water mark, not a per-request delta.
- Generated token count and finish reason, so outputs that stop at EOS can be distinguished from outputs hitting the token cap.
- Cold process-start-to-first-token and model-load durations for startup context.
- Host OS, architecture, core count, total memory, and relevant package versions.

The runner summarizes each prompt's 10 measured repetitions and pools the five prompt types at each length. It reports medians and nearest-rank p95s. With only 10 samples, nearest-rank p95 is the maximum observation; treat it as a coarse tail indicator, not a precise tail-latency estimate. The pooled-by-length summary has 50 observations.

`gpu_utilization_percent` is currently null because MLX LM does not expose a validated whole-GPU utilization percentage in this runner. Follow the optional collection procedure in [Apple Silicon telemetry](apple-silicon-telemetry.md); keep values from that procedure separate until their exact meaning and sampling behavior are verified on this Mac.

## Comparison discipline

Keep the Mac on the same power mode and charger state, close competing GPU-heavy work, and compare captures from the same machine, OS, model files, dependency versions, prompt suite, and output cap. Run the ordinary suite without telemetry for primary latency comparisons. Change one optimization variable at a time and rerun the unchanged suite. If fixtures, decoding settings, or measurement code change, identify the protocol version in the result and do not combine those numbers with prior captures as if they were identical tests.

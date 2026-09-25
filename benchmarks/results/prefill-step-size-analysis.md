# Prefill-step-size sweep

Date: 2026-09-25 (UTC)

## Protocol

The same Llama 3.2 1B Instruct 4-bit model and benchmark suite were run with MLX-LM prefill step sizes 512, 1,024, 2,048, and 4,096. Each suite used four input lengths (64, 512, 2,048, 4,096 tokens), five prompt types per length, one warmup and 10 measured repetitions per fixture, greedy sampling, up to 128 generated tokens, concurrency one, and a fresh prompt cache per request. Each suite therefore contains 200 measured generations. The `powermetrics` collector sampled at 1-second intervals during each run.

The tables report pooled-by-input-length statistics across all five prompt types (50 measured generations per length). p95 is nearest-rank. Timings below were collected while `powermetrics` was running; treat them as instrumented comparisons, not as clean timing-only results.

## Results

TTFT in seconds, shown as median [p95]:

| Prefill step | 64-token input | 512-token input | 2,048-token input | 4,096-token input |
|---:|---:|---:|---:|---:|
| 512 | 0.123 [0.131] | 0.330 [0.340] | 1.083 [1.092] | 2.249 [2.412] |
| 1,024 | 0.121 [0.128] | 0.332 [0.349] | 1.064 [1.082] | 2.178 [2.192] |
| 2,048 | 0.123 [0.134] | 0.335 [0.344] | 1.063 [1.076] | 2.161 [2.174] |
| 4,096 | 0.151 [0.309] | 0.418 [0.480] | 1.346 [8.744] | 2.723 [2.892] |

Prefill throughput in tokens/second, median [p95]:

| Prefill step | 64-token input | 512-token input | 2,048-token input | 4,096-token input |
|---:|---:|---:|---:|---:|
| 512 | 1,427 [1,490] | 2,066 [2,099] | 2,048 [2,062] | 1,895 [1,925] |
| 1,024 | 1,423 [1,500] | 2,035 [2,082] | 2,079 [2,093] | 1,955 [1,963] |
| 2,048 | 1,415 [1,497] | 2,062 [2,073] | 2,097 [2,107] | 1,974 [1,983] |
| 4,096 | 1,390 [1,484] | 1,705 [1,923] | 1,662 [2,038] | 1,571 [1,696] |

MLX peak memory in GB, median [p95]:

| Prefill step | 64-token input | 512-token input | 2,048-token input | 4,096-token input |
|---:|---:|---:|---:|---:|
| 512 | 0.833 [0.839] | 1.230 [1.230] | 1.345 [1.345] | 1.483 [1.483] |
| 1,024 | 0.833 [0.839] | 1.230 [1.230] | 1.564 [1.564] | 1.656 [1.656] |
| 2,048 | 0.833 [0.839] | 1.230 [1.230] | 1.542 [1.542] | 1.635 [1.635] |
| 4,096 | 0.824 [0.836] | 1.230 [1.230] | 1.542 [1.542] | 1.630 [1.630] |

## Telemetry observations

During the benchmark intervals, the telemetry contained 330, 321, 320, and 462 one-second samples for step sizes 512, 1,024, 2,048, and 4,096 respectively. Every sampled thermal-pressure state was `Nominal`.

| Prefill step | GPU HW active frequency, median (MHz) | GPU HW active residency, median (p95) | GPU Power, median (W) |
|---:|---:|---:|---:|
| 512 | 1,393 | 91.41% (98.98%) | 20.85 |
| 1,024 | 1,396 | 91.83% (99.37%) | 21.13 |
| 2,048 | 1,397 | 91.65% (99.56%) | 21.65 |
| 4,096 | 1,116 | 89.72% (100.00%) | 13.23 |

`powermetrics` emitted a per-process `GPU ms/s` column, but every task and `ALL_TASKS` value in these logs was `0.00`; it did not provide usable process-level GPU-time attribution. The device-level GPU hardware active residency, frequency, and power fields were present. The logs did not contain an internal die-temperature reading. Thermal pressure is categorical, not a temperature.

## Interpretation and next steps

- Keep the current 2,048 default for now. It had the best median prefill throughput at 2,048- and 4,096-token inputs and the lowest median TTFT at 4,096 tokens, while short-prompt latency differences among 512, 1,024, and 2,048 were small.
- Consider 512 as a memory-oriented alternative: its MLX peak was about 9% lower at 4,096-token prompts and 13% lower at 2,048-token prompts than the 2,048 setting, with roughly 4% and 2% lower median prefill throughput respectively.
- 1,024 showed no compelling advantage over 2,048 in this suite.
- 4,096 was slower across lengths. Its 2,048-token `rewrite` fixture had six of 10 runs with TTFT from 4.37 to 12.34 seconds, while the other four were about 1.36-1.40 seconds. This drives the pooled 2,048-input p95 of 8.744 seconds. The cause is not established; thermal pressure remained nominal. Treat that tail result as an anomaly to investigate and repeat before drawing a strong conclusion.
- These captures ran in ascending setting order and under telemetry instrumentation. Confirm any chosen setting with timing-only suites, and repeat/interleave settings to reduce order and transient-system effects. Do not compare these instrumented timings directly with the earlier uninstrumented baseline.

## Artifacts

The committed telemetry logs have had each `Running tasks` table removed to avoid publishing process names and PIDs. The GPU/thermal samples and their timestamps remain unchanged. The per-process `GPU ms/s` column was all zero before redaction.

| Prefill step | Benchmark JSON | powermetrics log |
|---:|---|---|
| 512 | [prefill-step-512-telemetry-20260925T060849Z.json](prefill-step-512-telemetry-20260925T060849Z.json) | [prefill-step-512-telemetry-20260925T060849Z.txt](prefill-step-512-telemetry-20260925T060849Z.txt) |
| 1,024 | [prefill-step-1024-telemetry-20260925T061710Z.json](prefill-step-1024-telemetry-20260925T061710Z.json) | [prefill-step-1024-telemetry-20260925T061710Z.txt](prefill-step-1024-telemetry-20260925T061710Z.txt) |
| 2,048 | [prefill-step-2048-telemetry-20260925T062407Z.json](prefill-step-2048-telemetry-20260925T062407Z.json) | [prefill-step-2048-telemetry-20260925T062407Z.txt](prefill-step-2048-telemetry-20260925T062407Z.txt) |
| 4,096 | [prefill-step-4096-telemetry-20260925T063134Z.json](prefill-step-4096-telemetry-20260925T063134Z.json) | [prefill-step-4096-telemetry-20260925T063134Z.txt](prefill-step-4096-telemetry-20260925T063134Z.txt) |

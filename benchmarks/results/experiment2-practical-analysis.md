# Experiment 2 practical-quality analysis

Date: 2026-09-26

## Decision

Adopt KV-prefix reuse as a performance candidate for this Mac and workload. It passed the practical quality, latency, and memory gates in clean timing runs. It is not worse than rejecting KV caching on the evidence collected so far.

This decision does **not** claim bit-for-bit determinism. The earlier protocol 1.0 diagnosis remains valid: finite-precision evaluation shape can change a near-tied generated token.

## Confirmation pair

- Control: `experiment2-control-practical-timing-20260926T190000Z.json`
- KV reuse: `experiment2-reuse-practical-confirmation-20260926T190500Z.json`
- Five fixtures, one warmup plus ten measured sessions per fixture
- Four turns per session, 220 total turns and 200 measured turns per arm
- Greedy decoding, 128-token cap, concurrency one, prefill step size 2,048

The confirmation treatment completed every turn. All required-fact checks passed, all outputs happened to match the fresh control token-for-token, and all 150 measured cache-hit turns were eligible for paired timing analysis.

| Metric | Result | Gate |
| --- | ---: | ---: |
| Median paired cache-hit TTFT improvement | 71.49% | at least 10% |
| 512-token-profile median paired TTFT improvement | 66.52% | descriptive |
| 2,048-token-profile median paired TTFT improvement | 88.83% | descriptive |
| Median paired session peak-MLX-memory change | -6.79% | no more than +10% |
| Required-fact losses versus control | 0 | 0 |
| Exact token mismatches in confirmation | 0 of 200 measured turns | diagnostic only |

Median cache-hit TTFT changed from 0.432 s to 0.145 s in the 512-token profile and from 1.404 s to 0.155 s in the 2,048-token profile. These are clean, uninstrumented timing results; no power or thermal claims are made yet.

## Repeatability and semantic review

The first completed practical treatment, `experiment2-reuse-practical-timing-20260926T000000Z.json`, used the older completed control and also passed the performance and automated factual screens. It showed ten measured exact mismatches, all representing the same distinct response change at repair-workshop turn 2.

Both answers said that the workshop cannot accept a device with a swollen battery. The reuse answer changed incidental advice from leaving the device at home to contacting the workshop. The added contact suggestion was not stated in the reference, but it did not contradict the required answer or materially reduce usefulness. It therefore passed the approved practical gate with a caveat. The same-day confirmation later produced no mismatch.

This variability is why exact-token equality should remain a diagnostic metric rather than a practical correctness requirement for this optimization.

## Limitations

- The result applies to this model, MLX-LM version, Mac, fixture suite, and single-user sequential-chat workload.
- The arms were run sequentially, not randomly interleaved per session, so slow environmental drift can remain. The large observed effect and separate successful practical capture reduce, but do not eliminate, that concern.
- The required-fact screen is narrow. Manual comparison checks cache-related degradation; it does not certify that every base-model answer is ideal.
- The memory metric is MLX's session peak measurement, not total system RAM or energy use.
- GPU power, frequency, per-process GPU time, and thermal pressure still require a separate matched `powermetrics` control/treatment pass.

## Next step

The optional matched telemetry comparison was skipped by user decision on 2026-10-03. A control-only telemetry capture was completed locally, but without a matched KV-reuse capture it cannot support any power, GPU-time, or thermal comparison and is intentionally excluded from Git. Experiment 2 is closed on its clean latency, memory, and practical-quality evidence; it makes no telemetry claim.

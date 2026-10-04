# Experiment 5 analysis: KV-cache quantization

Date: 2026-10-03

## Decision

Do not enable KV-cache quantization by default for the current Llama 3.2 1B
interactive-chat configuration. No tested arm satisfied all approved gates.

Post-prefill 8-bit quantization is a technically useful **research-only**
result: it preserves the control's tested long-context capabilities, reduces
retained cache and active memory, and appears to improve decode throughput.
However, it cannot reduce whole-request peak memory because that peak occurs
during the unquantized prompt-prefill phase. It therefore fails the approved
requirement of at least 10% lower MLX peak memory at 16K.

## Controlled setup

- Model: existing Llama 3.2 1B Instruct affine-4/group-64 weights.
- Runtime: MLX 0.32.2 and MLX-LM 0.31.3 in the project `.venv`.
- Contexts: 4,096, 8,192, and 16,384 prompt tokens.
- Prefill step size: 2,048.
- Greedy decoding, fresh request cache, concurrency one.
- KV arms: unquantized control, 8-bit/group-64, and 4-bit/group-64.
- Timing captures were clean runs without `powermetrics`.

The experiment introduced direct measurement of logical cache tensor bytes,
allocated cache bytes, cache type/layer count, layer offsets, active MLX memory,
and peak MLX memory.

## Measurement validation

The control measured exactly 32,768 logical KV bytes per cached token:

`16 layers × key/value × 8 KV heads × 64 head dimensions × 2 BF16 bytes`

All 16 layers switched to `QuantizedKVCache` in both quantized arms. Measured
bytes per cached token were 17,408 for 8-bit and 9,216 for 4-bit, matching the
expected scale/bias overhead and reductions of 46.875% and 71.875%.

Cache offsets were equal across all layers and matched prompt plus generated
tokens. This confirms that the byte savings are real cache compression rather
than missing or truncated cache state.

## Immediate quantization failure

MLX-LM's immediate path first constructs ordinary KV state, converts it, and
then performs later prefill chunks through a quantized cache. At 16K this lowers
retained memory but creates large conversion/dequantization temporaries during
prefill.

| 16K one-prompt smoke | Control | 8-bit immediate | 4-bit immediate |
|---|---:|---:|---:|
| Logical KV bytes | 537,329,664 | 285,456,384 (-46.88%) | 151,142,400 (-71.87%) |
| MLX active memory | 1.241 GB | 0.986 GB (-20.54%) | 0.850 GB (-31.52%) |
| MLX peak memory | 1.991 GB | 6.120 GB (+207.37%) | 6.842 GB (+243.62%) |
| TTFT | 12.366 s | 20.386 s (+64.86%) | 18.843 s (+52.38%) |

Both immediate arms fail the peak-memory and TTFT gates and were stopped.

## Post-prefill timing improvement

A separately labeled implementation kept the cache unquantized during prompt
prefill and set the quantization threshold to the actual prompt length. This
converts the cache once at the prompt boundary, before token-by-token decode.

At 16K, the one-prompt smoke produced:

| Metric | Control | 8-bit post-prefill | 4-bit post-prefill |
|---|---:|---:|---:|
| Logical KV bytes | 537,329,664 | 285,456,384 (-46.88%) | 151,077,888 (-71.88%) |
| MLX active memory | 1.241 GB | 0.986 GB (-20.54%) | 0.850 GB (-31.52%) |
| MLX peak memory | 1.991 GB | 1.991 GB (0.00%) | 1.991 GB (0.00%) |
| TTFT | 12.366 s | 12.758 s (+3.17%) | 12.874 s (+4.11%) |

Moving quantization to the prompt boundary removes the immediate path's peak
and latency explosion. It cannot lower whole-request peak memory, because the
unquantized prefill has already reached that peak before conversion.

## Long-context quality gate

Five deterministic quality fixtures placed required facts near the beginning,
middle, and end of 4K/8K/16K contexts. They covered exact recall, distributed
two-step lookup, a late conditional fact, structured extraction, and grounded
summary. Candidate quality was evaluated relative to the unquantized control,
because the 1B control itself cannot satisfy every difficult structured task.

- At 4K, control and post-prefill 8-bit both passed 11 of 22 checks and produced
  byte-identical output on every fixture.
- At 4K, post-prefill 4-bit passed only 9 of 22 checks: 81.8% of control, below
  the approved 95% threshold. It was rejected.
- Across 8K and 16K, control passed 32 of 44 checks and post-prefill 8-bit passed
  31 of 44, or 96.9% of control.
- Post-prefill 8-bit introduced no new critical case failure and preserved all
  focused fact-retrieval passes at every context length.

## Bounded 8-bit performance screen

The five quality tasks at each length also supplied a bounded, single-run
screen. These are medians across different deterministic tasks, not a repeated
full benchmark and not confirmation-quality speed claims.

| Context | TTFT change | Decode tok/s change | Direct KV bytes | Active MLX memory | Peak MLX memory |
|---:|---:|---:|---:|---:|---:|
| 4K | +0.38% | +15.79% | -46.88% | -7.88% | 0.00% |
| 8K | +0.13% | +26.26% | -46.88% | -13.26% | 0.00% |
| 16K | +0.11% | +21.43% | -46.88% | -20.54% | 0.00% |

The decode result is promising and consistent with lower memory traffic, but it
was not promoted to a confirmed speed claim because the arm already failed the
hard whole-request peak-memory gate.

## Gate result

| Arm | Cache-byte gate | Quality gate | Latency smoke | 16K peak-memory gate | Outcome |
|---|---|---|---|---|---|
| 8-bit immediate | Pass | Not advanced | Fail | Fail badly | Reject |
| 4-bit immediate | Pass | Not advanced | Fail | Fail badly | Reject |
| 8-bit post-prefill | Pass | Pass | Pass in bounded screen | Fail: 0% reduction | Research-only |
| 4-bit post-prefill | Pass | Fail | Pass in smoke | Fail: 0% reduction | Reject |

No candidate advanced to the three-repeat or ten-repeat performance captures,
because a repeated run cannot change the structural fact that post-prefill
quantization happens after the request's memory peak. No deployment setting was
changed.

## Raw evidence

- 4K immediate smoke: `experiment5-smoke-control-20261003.json`,
  `experiment5-smoke-8bit-20261003.json`, and
  `experiment5-smoke-4bit-20261003.json`.
- 16K immediate smoke: `experiment5-smoke-v1-16k-control-20261003.json`,
  `experiment5-smoke-v1-16k-8bit-20261003.json`, and
  `experiment5-smoke-v1-16k-4bit-20261003.json`.
- 16K post-prefill smoke:
  `experiment5-smoke-postprefill-16k-8bit-20261003.json` and
  `experiment5-smoke-postprefill-16k-4bit-20261003.json`.
- 4K quality screen: `experiment5-quality-smoke-control-4k-20261003.json`,
  `experiment5-quality-smoke-postprefill-8bit-4k-20261003.json`, and
  `experiment5-quality-smoke-postprefill-4bit-4k-20261003.json`.
- 8K/16K quality screen: `experiment5-quality-control-8k-16k-20261003.json`
  and `experiment5-quality-postprefill-8bit-8k-16k-20261003.json`.

## What we learned

KV-cache quantization is effective at reducing retained long-context state, and
8-bit quantization can preserve the tested capabilities. Placement matters:
quantizing during chunked prefill is harmful on this setup, while quantizing at
the prompt boundary avoids that penalty and may speed decode.

For this 1B model on a 16 GB Mac, retained KV memory is not the current limiting
resource. A future experiment could explicitly target **decode-phase active
memory and throughput** rather than whole-request peak memory, but that would be
a new success rule and should not be retroactively substituted here.

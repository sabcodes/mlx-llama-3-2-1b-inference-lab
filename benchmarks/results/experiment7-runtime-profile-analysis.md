# Experiment 7 analysis: runtime profiling

Current completion update (2026-10-20): see
[separate M2 phase and isolated KV diagnostic](experiment7-m2pro-phase-and-kv-analysis-20261020.md).
Pending states below describe the historical checkpoint.


Date: 2026-10-04

## Decision

The profiling pass is complete at the request-phase level, but it does **not**
yet justify writing a custom Metal kernel. Short prompts are dominated by token
generation; long prompts are dominated by prompt processing and first-token
work. Longer context also makes every later output token slower, so there are
two distinct optimization targets rather than one universal bottleneck.

Focused prefill and decode GPU traces were recorded successfully. Kernel-level
time attribution remains pending because this Mac currently has Apple Command
Line Tools, not the full Xcode Metal Debugger needed to replay the captures.

## Controlled setup

- Llama 3.2 1B Instruct affine-4/group-64 weights.
- Apple M2 Pro with 16 GiB unified memory.
- MLX 0.32.2 and MLX-LM 0.31.3 in the project `.venv`.
- Greedy generation, concurrency one, fresh KV cache for every request.
- Experiment 1's selected prefill step size of 2,048.
- Deterministic summarization prompts at 64, 510, 2,047, and 4,096 actual
  tokens.
- EOS suppressed to force exactly 128 generated tokens.
- One warmup plus ten measured runs per prompt length.
- Clean timing without `powermetrics` or Metal capture overhead.

Every measured run reached the 128-token length cap, and all ten output-token
hashes matched within each case.

## Phase results

| Prompt tokens | Median TTFT | TTFT share | Median decode time | Decode share | Total request | Decode tok/s | Inter-token p95 | MLX peak memory |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.118 s | 16.7% | 0.587 s | 83.3% | 0.707 s | 216.2 | 4.65 ms | 0.836 GB |
| 510 | 0.319 s | 33.7% | 0.630 s | 66.3% | 0.947 s | 201.7 | 4.96 ms | 1.229 GB |
| 2,047 | 1.035 s | 57.4% | 0.767 s | 42.6% | 1.801 s | 165.6 | 6.03 ms | 1.542 GB |
| 4,096 | 2.118 s | 69.3% | 0.937 s | 30.7% | 3.056 s | 135.6 | 7.41 ms | 1.635 GB |

TTFT means time through receipt of the first generated token. It includes
prompt prefill, first-token model work, and sampling; it is not a pure prefill
timer. Decode time covers the remaining 127 streamed token transitions.

## What this shows

For a short interactive request, decode is the main target: it consumes 83% of
the 64-token case and 66% of the 510-token case. For long histories, reducing
prompt/first-token work matters more: TTFT consumes 57% at 2,047 tokens and 69%
at 4,096 tokens.

Decode itself becomes context-sensitive. From 64 to 4,096 prompt tokens,
measured decode throughput falls from 216.2 to 135.6 tokens/s, a 37.3% drop,
while median per-run p95 inter-token latency rises from 4.65 to 7.41 ms, a
59.5% increase. This is consistent with each new token doing more work against
a larger KV history, but the phase timer alone cannot say which GPU kernel is
responsible.

Peak MLX memory rises from 0.836 GB to 1.635 GB across the same range. That is
well within this Mac's capacity, so memory is observable but is not the primary
reason to begin a custom-kernel project.

The ten-run timing was stable. The request-time p95 was less than 1% above the
median at every length, so the phase crossover is much larger than ordinary
run-to-run variation in this pass.

## Metal captures

Two steady-state captures were recorded at a 2,047-token prompt:

- Prefill/first-token capture: about 976 MB of logical trace contents.
- Sixteen-step decode capture: about 1.167 GB of logical trace contents.

The raw `.gputrace` directories are intentionally gitignored and remain local.
Capture wall time is not benchmark data because Metal recording adds heavy
overhead. MLX generation also pipelines one token ahead, so the trace boundaries
are approximate; the saved metadata states these limitations.

Without Xcode Metal Debugger, we cannot honestly rank quantized matmul,
attention, cache update, normalization, or other operation families by GPU
time. Coarse binary inspection did not expose trustworthy labels or durations.

## Gate result and next action

| Gate | Result |
|---|---|
| Reproducible phase split | Pass |
| Dominant phase at each workload | Pass |
| Focused GPU captures recorded | Pass |
| Dominant GPU operation family identified | Pending Xcode inspection |
| Custom kernel justified | No—not from current evidence |

Do not write a custom Metal kernel yet. First replay the two traces in Xcode's
Metal Debugger and determine whether one operation family has enough
end-to-end share for a plausible 10% request-time improvement. If no such
concentration exists, the better next experiment is a higher-level technique:
reduce/reuse long prompt work for long conversations, and target the
context-dependent decode path for short interactive chats.

## Evidence

- Phase timing: `experiment7-phase-profile-20261004.json`.
- Prefill capture metadata: `experiment7-prefill-capture-20261004.json`.
- Decode capture metadata: `experiment7-decode-capture-20261004.json`.
- Raw local-only traces: `experiment7-prefill-20261004.gputrace` and
  `experiment7-decode-20261004.gputrace`.

## Follow-up review, 2026-10-19

This file preserves the original M2 findings. The later
[review and completion record](experiment7-review-20261019.md) validates these
measurements and incorporates Arno's separately measured M4 phase profile and
reported Xcode shader-cost ranking. Local M2 operation attribution remains
unmeasured; M4 replay shares must not be transferred to this machine.

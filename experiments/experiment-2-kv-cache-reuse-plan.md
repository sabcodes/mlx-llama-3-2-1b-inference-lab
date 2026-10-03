# Experiment 2: multi-turn KV-prefix reuse

**Status:** complete. The original exact-token protocol correctly rejected cache reuse after a numerically sensitive token changed. The user then approved practical semantic equivalence as protocol 1.1. A fresh same-day control/treatment pair completed all 220 turns, preserved every required fact, produced no exact mismatches in the confirmation capture, and passed the latency and memory thresholds. An earlier practical treatment had one repeatable wording difference that passed manual review. Clean timing/quality supports KV reuse. The optional matched power/thermal comparison was intentionally skipped by the user on 2026-10-03; an unmatched control capture is retained locally and supports no comparative telemetry claim.

## User-approved design

- Five deterministic conversations, four user turns each, with 512-token and 2,048-token initial-prompt profiles, plus cache miss/session-reset checks.
- Generate assistant replies live in both arms. Compare generated token IDs; if they diverge, omit downstream turns from paired analysis (a warmup divergence stops the run).
- Greedy decoding, 128-token output cap, one warmup session and 10 measured repetitions per conversation.
- Predeclare a promising result as at least 10% median TTFT improvement on cache-hit turns, no more than 10% median session peak MLX memory increase, and no output mismatches.
- Capture clean timing-only control and reuse runs first. Capture telemetry separately for each arm afterward, because `powermetrics` instrumentation can affect timing.

## Protocol 1.1 practical-quality amendment

After diagnosing the exact-token mismatch, the user approved a more realistic rule: wording may differ when the answer remains equally correct and useful. Exact token equality remains recorded as diagnostic evidence but is no longer an automatic rejection.

The practical gate requires:

- no loss of the required answer fact when the control contains it;
- no material contradiction, instruction-following failure, or usefulness regression in manual review of every distinct changed answer;
- at least 10% median paired TTFT improvement on cache-hit turns; and
- no more than 10% median paired session peak-MLX-memory increase.

See `benchmarks/results/experiment2-practical-analysis.md` for the completed clean-run decision. This amendment does not erase the protocol 1.0 result: bit-for-bit generation is not guaranteed across different finite-precision evaluation shapes.

## Executive summary

Test whether reusing the already-computed key/value (KV) state for a conversation's unchanged token prefix makes single-user, multi-turn chat more responsive on the project's Apple-silicon Mac. Compare ordinary full-prefix recomputation with a cache-reuse path, using the same model, conversation fixtures, generation settings, and `prefill_step_size=2048` in both arms. This changes one variable: whether prior prefix computation is reused.

The existing Experiment 1 result is not itself evidence for this change: every request in that suite started with a fresh cache. Experiment 1's 2,048-token chunk size is retained as a fixed setting, not retested here. The primary outcome is per-turn time to first token (TTFT); memory growth and output equivalence are required guardrails.

## What is established vs. proposed

Established from the current repository:

- Reference setup: `mlx-community/Llama-3.2-1B-Instruct-4bit`, MLX-LM 0.31.3, MLX 0.32.2, macOS 15.6.1, Apple arm64, and a 16 GiB M2 Pro MacBook Pro. These versions describe the saved capture, not a promise that future runs use identical versions.
- The accepted diverse benchmark uses five task types (Q&A, rewrite, context question, extraction, summarization), four target input sizes (64, 512, 2,048, 4,096), greedy decoding, up to 128 new tokens, concurrency one, one warmup and 10 measured repetitions per fixture.
- Its ordinary request path starts with a fresh generation cache each time. It is a single-request benchmark, not a multi-turn conversation test.
- Experiment 1 retained `prefill_step_size=2048` as the current setting. The step-size report explicitly recommends clean timing-only confirmation and interleaved/repeated settings before stronger conclusions.
- Existing instrumentation reports TTFT, prefill/decode throughput, request time, MLX active/peak memory, and process peak RSS. Process peak RSS is a process-lifetime high-water mark, not per-request memory. The collected `powermetrics` logs do not supply usable process-attributed GPU time or internal temperature.

The following approved fixture profiles are generated from fixed, source-controlled reference notes; the runner records actual tokenizer-rendered lengths and fixture hashes. The longer profile uses deterministic numbered supplemental records to reach its token target, so treat it as a controlled long-context stress profile rather than natural conversational prose.

## Goal and hypothesis

**Goal:** determine whether caching and reusing the exact common token prefix across turns improves interactive multi-turn latency for this model and Mac, without unacceptable memory growth, correctness differences, or session-management complexity.

**Primary hypothesis:** for a continuation turn whose rendered token sequence shares a substantial prefix with the preceding conversation, reusing the cached KV state for that prefix will reduce median and p95 TTFT versus recomputing the entire conversation prefix. The size of the gain should increase as the reusable prefix grows. Decode tokens/second after the first token is not expected to materially improve.

**Null / failure outcomes:** no meaningful TTFT benefit; benefit only for unrealistically long prefixes; cache setup/bookkeeping erases the gain; memory grows unacceptably across a session; or token/output mismatches show that the compared contexts are not equivalent. Any of these is a valid result.

## Treatment and control

| Arm | Per-turn behavior | Purpose |
| --- | --- | --- |
| **Control: full recompute** | Render the complete current conversation with the model's chat template, then generate the next assistant turn with a fresh KV cache. | Represents the established no-reuse path. |
| **Treatment: prefix reuse** | Retain the session KV state and reuse it only for token IDs that are an exact prefix of the new fully rendered conversation; process only the unmatched suffix. Reset the cache at a new session. | Isolates reuse of prior prompt work. |

Both arms must use identical model files, chat template, tokenizer, temperature 0, output-token cap, stop behavior, serial execution, and `prefill_step_size=2048`. No cross-session sharing, unrelated-prompt cache, quantization change, model change, or concurrent requests. The exact supported MLX-LM cache API and its handling of chat-template boundaries must be checked against the installed version before coding; do not assume that a CLI prompt-cache feature and the direct Python API used by the benchmark runner behave identically.

Cache eligibility is determined from token IDs, not just message text. For every turn, record the full rendered input token count, reusable-prefix token count, suffix token count, and cache hit/miss/reset reason. If the new prompt is not an exact extension of the cached prefix, fall back to full recompute and label that turn rather than reusing a merely similar prefix.

## Workload and conversation fixtures

The existing five task types are useful inspiration but are not by themselves a multi-turn suite. Create a separate, versioned session fixture set so results are not mixed into the current fresh-cache baseline.

**Recommended starting fixture proposal for user verification:** five deterministic conversations spanning the existing task variety, each with four user turns. Give each session a stable system message and shared context, then have user turns ask related follow-up questions so rendered history grows naturally. Include both a moderate and a longer shared-prefix profile (provisionally about 512 and 2,048 prompt tokens before a selected continuation); token counts must be measured after chat-template rendering, not estimated from characters. Ensure the suite includes at least one short-prefix or cache-ineligible case as a sanity check, plus one cache reset/new-session boundary.

The conversation texts should be fixed in source-controlled fixtures (no generated/randomized prompt text during a run), safe for publication, and designed so answers can be checked for basic factual consistency. Avoid copying the long repeated filler context from the single-turn benchmark when it would create an artificial reuse ratio. User turn prompts should be the same in the paired arms. Use a fixed system prompt and deterministic greedy decoding. If earlier assistant messages are generated within the session, both arms must generate and append them under identical rules; if their generated token sequences diverge, flag the session as non-equivalent and do not treat later turns as a paired timing observation.

To preserve pairing, recommended run order is paired sessions: use the same fixture in both arms, with arm order alternated across fixtures/repetitions (AB/BA), or randomize a recorded arm order. Do not run every control session first and every treatment session second. Do not mix timings from the old single-turn suite into this analysis.

## Metrics

### Primary outcome

- **Per-turn TTFT (seconds):** start of the user's turn's generation call to first generated token. Compare paired treatment vs. control by conversation, turn index, actual rendered prompt length, and reusable-prefix length.
- **TTFT improvement:** `(control - treatment) / control`, reported as paired absolute milliseconds and percentage; report median and p95 across repetitions. Also report the distribution per turn position and context profile so a pooled result cannot hide where reuse helps.

### Required latency and workload metrics

- End-to-end turn/request duration and session wall time.
- Prefill tokens/second as returned by MLX-LM, with clear note that reused tokens may not be counted as newly prefetched; interpret alongside actual suffix tokens and TTFT.
- Decode tokens/second and generated-token count. These are secondary and mainly check for regressions.
- Rendered prompt token count, exact reusable-prefix token count, suffix tokens processed, cache hits/misses/resets, and cache-reuse fraction.
- Model load and first-session cold-start latency separately from warmed steady-state turns.

### Memory and system guardrails

- MLX active/peak memory for each turn and maximum active/peak memory per session; compare the treatment's growth with the control. Record cache length and, if the API exposes it, cache allocation/bytes. Do not infer cache bytes from token counts if the API does not expose them.
- Process peak RSS, labeled as a process-lifetime high-water metric. It cannot by itself identify a per-turn delta; compare separate fresh processes or report the limitation.
- Out-of-memory/errors, cache eviction/reset events, and any failed generation.
- Optional separate `powermetrics` diagnostic run: device-level GPU active residency/frequency/power and thermal pressure, if the user's already-authorized workflow is used. Do not treat all-device GPU power as process attribution; do not claim internal temperature if not reported. Keep these telemetry runs out of primary clean latency results.

### Correctness and equivalence guardrails

- For each paired turn, compare actual input token IDs and their longest common prefix. Record the prefix boundary and assert it is valid for treatment reuse.
- Compare generated token IDs when available; otherwise compare generated text and its SHA-256 hash. Greedy decoding should make this deterministic under equivalent state, but hash equality is a guardrail rather than a full semantic-quality evaluation.
- Record finish reason and whether the output hit the token cap. A mismatch should be investigated before interpreting speed data; mismatched turns and all downstream turns in that live session are excluded from paired speed claims and counted as correctness failures.
- Add a small, explicit answer check for fixture facts (or reviewed expected outputs) so a fast but contextually wrong continuation is not considered a win.

## Environment and run controls

- Use the same local model artifact, project-local `.venv`, Python/MLX/MLX-LM versions, macOS version, and code revision for both arms. Do not download/replace weights or update dependencies mid-experiment.
- Record model ID, local artifact identifier/hash if available, tokenizer/chat-template details, `prefill_step_size=2048`, sampling, max-new-token cap, stop tokens, fixture-set version, cache API/implementation version, and runner revision.
- One request at a time; no other benchmark or known GPU-heavy activity. Keep charger/power mode and machine conditions consistent. Note material deviations.
- Primary comparison runs are timing-only, without `powermetrics`. If thermal/GPU diagnostics are wanted, run a separate matched diagnostic pass after the clean runs and keep files clearly paired but analytically separate.
- One model load per process is acceptable for steady-state paired sessions, but capture first-session/cold behavior separately. Avoid measuring fixture construction/tokenization as inference latency; report it separately.
- Use one warmup per fixture/session and 10 measured repetitions as the initial recommendation, matching the existing suite. The same warmup/repetition rule must apply to both arms. Record every observation, including exceptions and partial runs; never silently delete outliers.
- To reduce order drift, alternate or randomize arm order in paired blocks with the order recorded. Keep run ordering deterministic after it is generated so analysis can be reproduced.

## Analysis and decision criteria

1. Validate input equivalence, cache-prefix correctness, and outputs before comparing speed. Correctness failure is a stop condition, not a performance tradeoff.
2. Summarize by fixture and turn position first, then by actual prompt-length/prefix-length bucket. Report raw observations, paired median difference, median percentage change, nearest-rank p95, and sample count. For 10 observations p95 equals the maximum; do not present it as a precise tail estimate.
3. Show cache benefit as a function of reused-prefix tokens and reused fraction. Include short-prefix/cache-miss cases as negative controls; there should be little/no advantage when no eligible prefix exists.
4. Compare treatment vs. control on memory high-water and session memory trend as well as latency. Check whether memory grows monotonically across turns or sessions and whether resets reclaim the cache. Do not compare process peak RSS as though it were a request-local measurement.
5. A provisional “promising” outcome is a repeatable reduction in median TTFT on cache-eligible continuation turns, with no output/input mismatch, no errors, and no material increase in end-of-session memory or decode/request latency. **The numeric minimum acceptable improvement and memory budget are user decisions**; select them before looking at the final results to avoid moving the goalposts.
6. If results are promising but noisy, repeat the finalist under the same protocol and alternate arm order. If the telemetry-instrumented sweep is used, confirm the result with timing-only runs. If benefit appears only at long prefixes, document the crossover rather than calling one global setting “best.”
7. Keep Experiment 1's existing report/artifacts unchanged. This experiment has a separate protocol and cannot be pooled with single-turn prompt-length statistics.

## Artifacts and suggested result structure

Save a plan-approved, versioned fixture file and one raw JSON result per arm/run under `benchmarks/results/`; add a separate analysis Markdown file after the experiment. Use new file names and do not overwrite Experiment 1 results. Do not put absolute home paths, usernames, process listings, or secrets in artifacts.

Suggested top-level JSON fields (schema proposal; runner implementation may refine names):

```json
{
  "schema_version": 1,
  "experiment": "experiment-2-kv-prefix-reuse",
  "protocol_version": "1.0",
  "started_at_utc": "...",
  "git_revision": "...",
  "arm": "control-full-recompute | treatment-prefix-reuse",
  "model": {"id": "...", "artifact_hash": "..."},
  "environment": {"macos": "...", "architecture": "...", "mlx_lm": "...", "mlx": "..."},
  "settings": {
    "prefill_step_size": 2048,
    "sampling": "greedy",
    "max_new_tokens": 128,
    "concurrency": 1,
    "warmups_per_session": 1,
    "measured_repetitions": 10,
    "telemetry": "off"
  },
  "fixture_set": {"id": "...", "sha256": "..."},
  "sessions": [{
    "fixture_id": "...",
    "repetition": 1,
    "arm_order": "...",
    "turns": [{
      "turn_index": 1,
      "input_tokens": 0,
      "reused_prefix_tokens": 0,
      "suffix_tokens": 0,
      "cache_status": "hit | miss | reset | fallback",
      "ttft_seconds": 0.0,
      "request_seconds": 0.0,
      "prefill_tokens_per_second": 0.0,
      "decode_tokens_per_second": 0.0,
      "generated_tokens": 0,
      "finish_reason": "...",
      "mlx_active_memory_bytes": 0,
      "mlx_peak_memory_bytes": 0,
      "output_sha256": "...",
      "input_prefix_verified": true,
      "output_equivalent": true
    }]
  }],
  "summary": {},
  "completed_at_utc": "..."
}
```

Store raw per-run data as the source of truth; calculate summaries from it. If the cache implementation cannot expose a metric, write `null` with an explanation instead of estimating it. Consider aggregating paired arms in an analysis file keyed by fixture, repetition, and turn index rather than duplicating derived numbers into raw captures.

## Execution checklist after approval

1. Confirm the five workload decisions above, especially live vs. replayed assistant history, history lengths, output cap, and repeat budget.
2. Inspect the exact installed MLX-LM version's supported Python cache/prefix-reuse interfaces; document API choice and invalidation semantics before coding.
3. Define a separate multi-turn fixture format and validator. Freeze/version fixtures and compute a content hash.
4. Extend or add a runner with explicit control/treatment arms, precise token-prefix validation, per-turn metrics, session-boundary resets, deterministic arm order, partial-result persistence, and JSON schema validation. Keep Experiment 1's runner defaults and data unchanged.
5. Run a small correctness-only smoke test first (not part of reported timing). Verify same token contexts and outputs for both arms and expected behavior on cache miss/reset.
6. Capture clean timing-only paired runs at `prefill_step_size=2048`, with the approved warmup/repeat count. Retain raw artifacts even if a session fails; label partials and exclude them from completed comparisons.
7. Validate every completed result for matching fixtures/turns, cache counters, repetitions, summaries, timestamps, and parseable JSON. Scan artifacts for sensitive local paths before sharing/committing.
8. Analyze paired TTFT, latency, memory, and correctness; write an experiment report with limitations and a decision to adopt, reject, or repeat.
9. If requested, collect a separate matched telemetry pass using the project's telemetry instructions; require the user to start the `sudo` collector in their own Terminal and confirm it is running before starting the benchmark.

## Risks and limitations

- **Chat-template token-prefix instability:** adding a message may alter special tokens at the former end of the rendered conversation. Reuse only the exact verified common token prefix and handle cache truncation/rebuild safely.
- **Wrong or stale KV state:** incorrect cache lifecycle can silently condition output on a different conversation. Session isolation, token-prefix checks, deterministic tests, and new-session reset are mandatory.
- **API/version mismatch:** MLX-LM public CLI flags do not guarantee an equivalent direct-Python API. Verify the installed pinned version; do not invent or assume a method name.
- **Biased workload:** long stable system prompts maximize apparent benefits; ordinary chat may have a smaller reusable fraction. Report absolute reused tokens and fraction, and include short-prefix/miss controls.
- **Memory accumulation:** cache reuse retains KV state across a conversation. Measure full-session high-water/growth and test cleanup/reset; speed is not a win if memory grows without a usable bound.
- **Timing drift and noise:** thermal/power state, background work, and run order affect measurements. Pair/alternate arms; preserve raw data; repeat finalists; avoid telemetry in primary timing runs.
- **Metric ambiguity:** library prefill tokens/second may refer only to newly processed suffix tokens in treatment. TTFT and explicit suffix/reused token counts are more directly interpretable.
- **p95 resolution:** with only 10 repetitions per fixture, nearest-rank p95 is the maximum and not a stable tail estimate.
- **Performance does not prove generality:** conclusions apply to the tested Mac, OS/runtime versions, model, cache path, and conversation profile only.

## Remaining pre-run checks

- The short-context smoke test matched token-for-token, but the 2,048-token repair-workshop fixture diverged on turn 2 during warmup. The cache reported a 2,129-token prefix and 19-token suffix for a 2,148-token prompt; its logical state and stored key both measured 2,230 tokens after generation. Re-rendered control and treatment turn-2 inputs were verified identical, and the preceding assistant output IDs matched. The first output difference was at generated token index 53 (zero-based), with 84 control tokens versus 82 treatment tokens.
- A separate diagnostic rebuilt the prefix cache without the LRU matcher, using the cached path's exact evaluation segment schedule. It matched the saved reuse output exactly, while a fresh full-prompt evaluation matched the saved control. At the first differing position, the control favored its token by only `0.015625` log-probability; the segmented path produced a tie. This rules out a wrong prefix/stale cache and identifies finite-precision sensitivity to different batching schedules as the cause. See `benchmarks/results/experiment2-cache-equivalence-diagnosis.md`.
- The full-recompute timing baseline completed: 5 fixtures, one warmup and 10 measured repetitions per fixture, 200 measured turns. The reuse capture stopped at the long-context warmup mismatch and remains partial; do not compare its preceding observations as a completed result.
- Keep baseline and treatment measurement modes matched: timing-only control JSON is compared only with timing-only reuse JSON; telemetry-instrumented control and reuse captures form a separate matched diagnostic pair. Telemetry is deferred until the correctness guardrail is resolved.
- Keep all raw JSON files and mark interrupted runs partial. Do not compare telemetry-instrumented timings with clean primary results.

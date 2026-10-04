# Experiment 6: speculative decoding with a tokenizer-compatible draft

## Plain-language question

Can a draft model guess several future tokens cheaply enough that Llama 3.2 1B
can verify them together and finish an answer faster than ordinary one-token-at-
a-time decoding?

This is an inference-only experiment. It does not train or fine-tune either
model and it does not change the target model's stored weights.

## Confirmed scope

- Hardware: the current M2 Pro MacBook Pro with 16 GB unified memory.
- Target: the existing Llama 3.2 1B Instruct affine-4/group-64 checkpoint.
- Initial draft: the existing Mixed-3/6/group-64 artifact from Experiment 4.
- Runtime: project-local `.venv`, MLX 0.32.2, MLX-LM 0.31.3.
- Use case: persistent, single-user interactive chat.
- Greedy generation, concurrency one, and prefill step size 2,048.
- Draft counts: 1, 2, 3, and 4 proposed tokens per verification round.
- Ordinary timing remains separate from optional privileged telemetry.

The target and draft tokenizer files are byte-identical. The draft is still the
same 16-layer, 1B architecture and is only 14.5% smaller in stored weights. It
is therefore a controlled feasibility probe, not an ideal small draft. Failure
of this pairing does not disprove speculative decoding with a genuinely small,
purpose-built draft.

## Known correctness risk

MLX-LM 0.31.3 has open reports of speculative decoding diverging from ordinary
greedy output. Speculative decoding is intended to change the schedule used to
obtain target-model tokens, not the target result. Exact token equality is
therefore a hard Stage-0 gate rather than an optional quality preference.

## Arms

1. Target-only deployment control in a fresh process.
2. Both models resident with zero drafted tokens, as a mechanism/residency
   control.
3. Speculative decoding with draft counts 1, 2, 3, and 4.

The deployment control must not load the draft model. Memory and startup from a
process containing both models cannot be presented as target-only cost.

## Stage 0: mandatory correctness probe

Use the five baseline prompt types at approximately 512 prompt tokens and a
128-token cap, plus two deterministic prompts forced to reach a 256-token cap.
For every draft count, record:

- exact target and speculative token IDs, text hashes, EOS position, generated
  length, and finish reason;
- first differing token position and values;
- target- and draft-cache layer offsets and internal consistency;
- emitted accepted-draft tokens, target-produced tokens, verification rounds,
  and reconstructed proposal opportunities;
- active and peak MLX memory as diagnostic context only.

Reject an individual draft count after any token mismatch, inconsistent cache,
crash, or incorrect EOS/token-cap behavior. Do not performance-test a rejected
arm. Stop the whole mechanism if every draft count fails.

## Stage 1: bounded mechanism screen

For surviving counts, compare the deployment control, zero-draft mechanism
control, and speculative arms at 64, 512, and 4,096 prompt tokens with both 32-
and 128-token caps. Use one warmup and three measured repetitions per fixture.

Advance only candidates that remain exactly equivalent, show at least a
plausible 5% median decode improvement at the 128-token cap, do not regress
median request time by more than 10%, and remain inside the memory guardrails.

## Stage 2: full and confirmation captures

For surviving arms, run all five prompt types at 64, 512, 2,048, and 4,096
tokens with 32- and 128-token caps, one warmup, and ten measured repetitions.
Use reversed or interleaved arm order to expose order and thermal bias. Report
paired per-fixture ratios, pooled medians, and nearest-rank p95. A selected
winner must reproduce in a second full capture.

## Metrics

- exact token equality, first divergence, EOS position, and finish reason;
- TTFT, decode tokens/second, request duration, and actual generated tokens;
- accepted draft tokens, proposed-token opportunities, accepted tokens per
  verification round, verification rounds, and emitted draft-token fraction;
- target load time, draft load time, cold process latency, active MLX memory,
  peak MLX memory, process peak RSS, and target/draft cache bytes where useful;
- fixed factual, formatting, and instruction-following checks for any survivor.

Acceptance is diagnostic, not a success criterion by itself. A high-acceptance
draft can still lose if producing its guesses costs too much.

## Approved decision rule

### Default optimization

All of the following must pass:

- 100% exact greedy-token equality on every deterministic fixture;
- at least 20% faster median decoding for 128-token workloads;
- at least 10% lower median total request time for 128-token workloads;
- no more than 5% slower median request time for 32-token workloads;
- median TTFT no more than 5% slower and p95 TTFT no more than 10% slower;
- no new factual, formatting, or instruction-following failure;
- peak MLX memory no more than 50% higher, no more than 0.8 GB higher in
  absolute terms, and below 3 GB total;
- no important per-length regression hidden by pooled results; and
- the win reproduces in a second full capture.

Report model-load and cold-process costs separately. A material cold-start
regression prevents an unconditional default recommendation even if a warm,
persistent chat session wins.

### Research-only

Use this label for a technically informative arm that produces a 10–20% decode
improvement, helps only long answers, uses 50–75% more peak memory, regresses
cold/short requests, or produces meaning-preserving but non-identical output.
Non-identical output must be labeled behavior-changing and cannot be called a
faithful acceleration.

### Reject

Reject an arm for unexplained token divergence, a new critical quality or
format failure, less than 10% end-to-end improvement on 128-token workloads,
more than 10% TTFT or short-response regression, more than 75% additional peak
memory, instability, or no repeatable speed advantage.

## Execution checklist

- [x] Confirm target/draft tokenizer and architecture compatibility.
- [x] Implement and validate the Stage-0 correctness probe.
- [x] Run Stage 0 for draft counts 1–4.
- [x] Apply the Stage-0 stop gate before performance work: every count passed
  all seven exact-token cases, including both 256-token probes.
- [x] Run the bounded Stage-1 screen for survivors: target-only, zero-draft,
  and draft counts 1–4 completed in separate processes.
- [x] Apply the Stage-1 stop gate: no arm showed a plausible speed route, so
  the full and confirmation captures were not run.
- [x] Write the decision, limitations, and raw-result paths.
- [x] Security-scan staged files before committing and pushing.

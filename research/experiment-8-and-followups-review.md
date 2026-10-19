# Experiment 8 and follow-up methods: M4 Pro review

Date: 2026-10-05 PDT

Status: design review only. No Experiment 8 arm has been run.

## Recommendation

Proceed with a bounded **3B target / 1B draft speculative-decoding study** only
as a new M4 Pro deployment question. The repository's current decision target
is a 1B model on M2 Pro; moving both machine and target model makes Experiment
6 an unsuitable speed baseline. Record a fresh M4 Pro 3B target-only baseline,
and compare all speculative arms only with that baseline. Keep the M4 Pro 1B
profile as a bridge, not a speedup denominator.

The hypothesis is that the 1B draft can pay for its serial proposals, second
prefill, second cache, and verification against the slower 3B target. Acceptance
rate alone does not establish a speedup. The [speculative-decoding method](https://arxiv.org/abs/2211.17192)
and [MLX-LM 0.31.3 implementation](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py) define
the relevant behavior.

## Experiment 8 gates

1. **Pin artifacts and semantics.** Freeze both model revisions, weight hashes,
   tokenizer and chat-template behavior, the rendered prompt-token IDs, MLX and
   MLX-LM versions, macOS, power conditions, and 3B target quality fixtures.
   Check token ID and special-token mapping, not just vocabulary size. The
   published [3B config](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit/raw/main/config.json)
   has 28 layers and the local 1B has 16. The current Experiment 6 runner
   rejects unequal structural signatures, so that repo-only check must be
   replaced with correct target/draft cache-shape validation before this pair
   can run.
2. **Stage 0, correctness.** Compare exact greedy target token IDs, output
   length, EOS position and finish behavior for short, long, forced 256-token,
   and multi-turn requests. Validate independent cache growth and reuse for
   both models. Stop immediately on divergence, invalid cache, or crash. Exact
   matches preserve target output on tested greedy cases; do not extend that
   claim to sampling without a separate test.
3. **Stage 1, bounded screen.** In separate clean processes, measure 3B
   target-only, 3B plus resident 1B with zero proposals, proposal counts 1–4,
   and standalone 1B. Record draft-only token cost, actual proposal counts,
   accepted tokens per verification, load/cold-start cost, TTFT, decode,
   end-to-end request time, peak memory, headroom, and swap. Stop if no count
   has a plausible 5% decode gain, or if overhead makes the default gates
   unattainable.
4. **Full decision.** Require at least 20% lower median 128-token decode time
   and 10% lower median short-prompt request time against M4 Pro 3B target-only.
   Allow at most 5% regression for 32-token requests and median TTFT, and 10%
   for p95 TTFT. Reject important per-length regressions. Set a fresh memory
   ceiling for the 24 GiB M4 Pro rather than inheriting Experiment 6's 1B/M2
   limit. Confirm a winner in a second clean capture with varied prompt types,
   lengths, and interleaved or reversed arm order. Ten samples give a coarse
   p95; expand only if that tail gate is close.

The 1B M4 Pro Experiment 7 profile is 85% decode at 64 prompt tokens but 29%
decode at 4,096. Its [Metal replay](../benchmarks/results/experiment7-m4pro-runtime-profile-analysis.md)
identified quantized matrix work as the main GPU shader cost for the tested
1B prompt. These observations warn about duplicated long-prompt prefill, but
do not predict the unmeasured 3B phase split.

## Other experiments, ranked

| Priority | Method | Why it is worth testing | Stop rule |
|---|---|---|---|
| 1 | Focused 4-bit matrix-path comparison | Xcode replay identified affine quantized matrix multiplication as 90.9% of prefill shader cost and matrix-vector work as 75.7% of decode shader cost for a 2,047-token 1B prompt. First compare available MLX/runtime choices; design a minimal kernel only if a specific gap remains. Use [Apple's Metal profiling workflow](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app) and [MLX Metal Debugger guidance](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html). | No plausible path to 10% end-to-end gain, or a candidate wins only in a microbenchmark. |
| 2 | Long-chat context budget and selective history retention | Preserve system instructions, recent turns, and required facts while reducing old attention/KV work. This is distinct from exact-prefix reuse and KV quantization. [StreamingLLM](https://arxiv.org/abs/2309.17453) is a research starting point, not proof that its policy will preserve this project's tasks. | Any required-history fact or format fails, or full-turn gain is under 10%. |
| 3 | Cross-runtime MLX-LM versus llama.cpp Metal | A whole-runtime comparison may expose a better scheduler, kernel, or cache path. Start with equivalent precision from one source checkpoint, then compare practical 4-bit artifacts with quality gates. [`llama-bench`](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) is useful for screening, but final decisions need the same request harness. | Less than 10% end-to-end gain after matching prompt and output semantics, or unacceptable quality/memory cost. |
| 4 | Greedy-only decode path | For a true greedy workload, `argmax(logits)` is unchanged by subtracting the shared `logsumexp(logits)`. The pinned [MLX-LM generation path](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py) computes normalized log probabilities. Measure whether skipping unused work matters. | Any token mismatch or less than 3–5% full-request gain. |
| 5 | KV allocation-step/preallocation sweep at 8–16K context | The pinned [MLX-LM cache](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py) grows in steps; fewer reallocations might help longer sessions, with a memory cost. Verify the Llama cache actually uses the chosen mechanism before measuring. | Less than 5% full-request gain or excessive peak memory. |

For close decisions, use randomized paired control/candidate blocks and a clean
confirmation pass to reduce clock, thermal, and background-load drift. The
[Google Benchmark guide](https://google.github.io/benchmark/user_guide.html)
documents the general repeated-measurement principles. Record exact prompt
token hashes and outcome-level latency; kernel or token/s changes alone do not
select a default.

# Experiment 3: explicit MLX compiled decode

**Status:** complete at Stage A; explicit whole-step compiled decode rejected for the pinned MLX 0.32.2 / MLX-LM 0.31.3 stack. Shapeless compilation could not infer the growing KV-cache slice shapes. Shape-specific compilation preserved tokens and cache length but retraced once per token, slowed warm decode by 36.9% in the bounded probe, made the first token much slower, and increased peak MLX memory by 10.6%. The predeclared stop rule prevented a wasteful full-suite run.

## Question

Can an explicit `mx.compile` path reduce steady-state, single-token decode overhead for Llama 3.2 1B on this M2 Pro without making interactive requests slower overall, increasing memory unacceptably, or degrading answer quality?

This is inference engineering. It does not train, fine-tune, or modify model weights.

## Evidence from the pinned runtime

- The project uses MLX 0.32.2 and MLX-LM 0.31.3.
- The installed `mlx_lm.generate.generate_step` calls the model eagerly for prompt chunks and for each decoded token; no explicit `mx.compile` call exists in the installed MLX-LM package.
- `mx.compile` caches a traced computation and can fuse work, but the first evaluated call pays tracing/compilation cost.
- MLX compilation must explicitly capture changing state. Decode mutates every layer's KV cache.
- Input shape, dtype, input count, and uncaptured Python constants can create distinct compiled traces. The cache offset changes every token, so the implementation must prove it is not recompiling or retaining a new trace at every position.

Primary references:

- MLX compilation guide: https://github.com/ml-explore/mlx/blob/main/docs/src/usage/compile.rst
- Installed MLX-LM generation source: `.venv/lib/python3.14/site-packages/mlx_lm/generate.py`
- Installed MLX-LM cache source: `.venv/lib/python3.14/site-packages/mlx_lm/models/cache.py`

## Proposed two-stage experiment

### Stage A: bounded feasibility probe

Use the local model and project `.venv`. Compare eager and compiled single-token decode after the same eager prefill. Limit the probe to a small fixed prompt and output so a bad compile strategy cannot create unbounded traces.

The probe must verify:

1. identical prompt token IDs and cache starting state;
2. generated token equality or practical semantic equivalence under the same Experiment 2 rule;
3. cache length/state advances correctly;
4. first compile cost, subsequent per-token latency, active/peak MLX memory, and process RSS;
5. whether changing cache offset or token position triggers repeated compilation or persistent memory growth; and
6. cleanup/reset behavior after the compiled decoder is discarded.

Stop before the full suite if compilation fails, recompiles per token, grows memory without a usable bound, corrupts cache state, or loses answer quality.

### Stage B: controlled benchmark

Only if Stage A passes, run the same five prompt types at 64, 512, 2,048, and 4,096 input tokens, with one warmup and ten measured repetitions, greedy decoding, a 128-token cap, concurrency one, and prefill step size 2,048.

Keep prompt prefill eager in both arms so the one changed variable is explicit compilation of the single-token decode path. Record compilation/startup separately from steady-state decoding.

Compare:

- current eager decode control;
- explicit compiled-decode treatment;
- first-request end-to-end latency including compile cost;
- warm steady-state decode tokens/second and per-token latency;
- TTFT, request duration, output tokens and quality;
- MLX active/peak memory and process peak RSS; and
- break-even point: how many generated tokens or requests are required to repay compilation cost.

If the isolated treatment wins, run a smaller confirmation on the adopted cumulative chat stack (prefill step size 2,048 plus KV-prefix reuse). Do not silently treat isolated and cumulative results as the same workload.

## Proposed success gate requiring user confirmation

Adopt compiled decode only if:

- median steady-state decode throughput improves by at least 10%;
- median end-to-end request time does not regress after including the user-visible compilation strategy;
- median peak MLX memory increases by no more than 10%;
- there is no material factual, instruction-following, or usefulness regression; and
- the implementation does not compile/retain an unbounded number of traces as token position or prompt length changes.

Report cold compile cost and break-even even if the steady-state threshold passes. A warm-only speedup that does not repay its compilation cost in realistic 128-token interactive answers is a rejection for this project's current use case.

## Decision points

Before implementation, confirm:

1. the proposed 10% decode-throughput and 10% memory thresholds; and
2. the two-stage structure: isolated single-turn suite first, then a smaller cumulative KV-cache chat confirmation only if compilation wins.

The user approved proceeding on 2026-10-03. Stage A failed its feasibility gate, so Stage B and the cumulative KV-cache confirmation were not run. See `benchmarks/results/experiment3-compiled-decode-analysis.md`.

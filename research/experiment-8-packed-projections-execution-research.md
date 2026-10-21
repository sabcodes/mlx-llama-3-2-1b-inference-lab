# Experiment 8 execution research and adversarial mechanism review

Checked: 2026-10-20.
Scope: local M2 Pro / 16 GB; existing Llama 3.2 1B Instruct affine-4/group-64 checkpoint.
Status: pre-implementation research. This report contains no benchmark result or successful correctness claim.

## Recommendation

Execute the preregistered C/Q/G/QG row-packing comparison with the pinned environment, after covering actual quantized dispatch boundaries and checking downstream layout effects. Current research supports investigating dispatch and data-layout overhead; it does not predict a local speedup. A clean rejection remains a valid result. Keep unconventional alternatives in a bounded follow-up register so they cannot become post-hoc replacement arms.

The governing local proposal is [the plan](../experiments/experiment-8-packed-projections-plan.md) and [the specification](../experiments/experiment-8-packed-projections-spec.md). This review inspected their proposed design before implementation. Execution authorization now comes from the current user request; historical design-only wording in those documents is not a new permission barrier.

## Pinned environment versus current technology

| Component | Installed/project pin inspected | Current official release checked | Consequence |
|---|---|---|---|
| MLX and MLX Metal | 0.32.2 | 0.32.4 | Keep the experiment pinned; an upgrade changes the control environment. |
| MLX-LM | 0.31.3 | 0.32.2 | Installed Llama/generation/cache source governs implementation. |
| Hardware | M2 Pro, 16 GB | Research also covers newer Apple generations | No cross-machine result is a local speed denominator. |
| Weight format | Existing affine 4-bit, group 64 | New quantized types and runtime formats exist | Preserve packed bytes; no re-quantization or conversion in this experiment. |

The installed versions were read from local package directory names and Python source, without importing MLX or executing GPU operations. Official [MLX 0.32.4 release notes](https://github.com/ml-explore/mlx/releases/tag/v0.32.4) and [MLX-LM releases](https://github.com/ml-explore/mlx-lm/releases) establish the current release distinction. MLX 0.32.4 includes view, stride, contiguity, split-span, and evaluation-cleanup fixes; this makes precise layout assertions useful, but does not prove a defect in the pinned path or justify silently upgrading it. The [0.32.3 release notes](https://github.com/ml-explore/mlx/releases/tag/v0.32.3) also include targeted attention fixes; they are not evidence of a generic projection-packing improvement.

## Mechanism and source-derived hazards

The installed Llama source computes Q/K/V separately from the same attention input and gate/up separately from the same MLP input. It uses `swiglu(gate, up)`, then the existing down projection. The local dimensions in the proposal are Q/K/V 2,048/512/512 and gate/up 8,192/8,192. [Pinned Llama source](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/llama.py).

Packing output rows preserves the quantized values only if weight, scales, and affine quantization biases are concatenated in identical row order, with matching bits, group size, mode, orientation, and dtype. Ordinary linear bias is a separate field. Installed `QuantizedLinear.__init__` creates random dense weights and quantizes them, so calling that constructor for a large packed layer merely to overwrite its arrays adds avoidable work and allocations. A minimal experiment-local module can carry the existing arrays and attributes. [Pinned quantized layers](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/quantized.py).

Pinned Metal dispatch depends on input rows, input width, output width, and GPU architecture. Its vector/matrix switch includes thresholds such as 10, 12, and 18; changed output width can change the selected threshold. Its quantized matmul also ensures the last two input dimensions are contiguous, inserting copies otherwise. Add rows 9/10/11, 11/12/13, and 17/18/19, or exact device-derived limits plus/minus one, to the numerical checks. Inspect output slices as well as packed parameter layout: a slice of a larger output can retain a larger row stride. Affine `qmv_wide` is gated to newer architecture generations, so its existence alone is not evidence it benefits M2. These observations come from [pinned Metal quantized dispatch](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/backend/metal/quantized.cpp); actual runtime dispatch/copies remain unmeasured.

Engineering inferences to verify:

- Concatenating different projection weights does not reduce total weight bytes or arithmetic. Benefits could come from fewer calls, shared input reads, and launch geometry.
- Output splitting, changed kernel selection, extra copies, and longer activation lifetimes can consume those benefits.
- Three projection proxies must not each evaluate the packed matmul. Replace the attention call experiment-locally so the shared projection is evaluated once, preserving RoPE, mask forwarding, cache updates, SDPA, and output projection semantics.
- Retained source modules, closures, and lazy graphs can preserve duplicate weights. Materialize layerwise packing before releasing obsolete references and verify retained parameter bytes.
- Identical packed bytes do not imply identical floating-point outputs: a different shape may select a different accumulation path. Numerical tolerance must never excuse a token/EOS mismatch under this specification.
- Fewer logical projection calls do not prove fewer Metal dispatches, higher bandwidth, or full-request speedup.

## Recent evidence and its limits

[MLX issue #4265](https://github.com/ml-explore/mlx/issues/4265) reports a small-row quantized-matmul problem on M3 Ultra and describes an important measurement reversal: an independently enqueued microbenchmark favored a kernel that slowed the actual model, while a dependency-chained benchmark exposed its latency loss. This is contributor-reported evidence, not an independent reproduction on this machine. Keep layer microbenchmarks diagnostic and use dependent/full-model requests for selection.

[MLX-LM PR #1922](https://github.com/ml-explore/mlx-lm/pull/1922), opened in September 2026 and still open when checked, targets reuse across 4–8 simultaneous rows and reports single-stream parity. It is useful evidence for a future batch-serving or multi-token verification question, not evidence of a concurrency-one decode improvement here. Do not import its wrapper/kernel into this controlled experiment.

The July [BaseRT paper](https://arxiv.org/html/2607.00501v1) motivates fused operators, preallocated decode scratch, and hardware-adaptive dispatch. Its reported evaluation uses M3 and M4 Pro, so its claimed gains do not establish M2 benefit. The [current BaseRT repository](https://github.com/basecompute/baseRT) converts checkpoints into `.base`; the engine is distributed as a proprietary prebuilt binary under a separate license. A future comparison would require matching weights, tokenizer, prompt rendering, output behavior, and request timing, plus an agreed install/conversion storage budget. The paper's runtime-level gain does not isolate row packing as its cause.

Apple's [WWDC26 TensorOps guidance](https://developer.apple.com/videos/play/wwdc2026/330/) describes portable tensor operations and expanded quantized data types while identifying the GPU Neural Accelerator as new M5 hardware. The [Metal capability table](https://developer.apple.com/metal/capabilities/) lists M2 Metal 4 support, but API support is not M5 accelerator availability. A new TensorOps kernel needs exact device/OS/SDK verification and an independent implementation experiment.

Two unconventional directions deserve a research backlog. [Open-TQ-Metal](https://arxiv.org/abs/2604.16957) studies compressed-domain int4 attention on substantially larger models and very long contexts. [When Quantization Is Free](https://arxiv.org/abs/2605.05699) studies fused transformed KV quantization on Gemma/Qwen through a Hugging Face cache interface. Neither demonstrates transfer to this Llama 1B MLX workload; transformed caches alter numerical behavior and require task-quality validation. Attention speedups at extreme context cannot be treated as full-request improvements at 64–4,096 tokens.

## Feasibility and bounded exploration register

| Priority | Approach | Feasible with current artifacts? | Appropriate experiment boundary | Main falsifier/limitation |
|---|---|---|---|---|
| 1 | C/Q/G/QG row packing | Yes; existing API and weights | Current Experiment 8, after correctness/layout checks | No clean short-request improvement or any correctness/memory failure |
| 2 | Packing only during decode or only during prefill | Technically plausible without a download | New preregistered arms after the current decision | Branching/setup overhead or loss on required lengths; no post-hoc rescue |
| 2 | Contiguous versus strided packed-output splits | Plausible with existing API | Current diagnostics; any changed timing arm requires registration | Copies cost more than the packed call saves |
| 3 | Gate/up projection fused with SwiGLU; QKV fused with RoPE/cache write | Requires new custom Metal code | Separate kernel experiment; preserve baseline semantics | Numerical mismatch, register/scratch pressure, no full-request win |
| 3 | Greedy sampling path without unused normalization | Existing source/API, separate code change | Independent overhead experiment | Token mismatch or negligible request benefit |
| 4 | KV allocation/preallocation and scratch reuse | Requires cache/lifecycle changes | Separate experiment with long-session cases and memory ceiling | Startup/capacity cost exceeds latency gain |
| 4 | Selective dequantization for prefill | Same packed values can be dequantized, but adds substantial resident memory and another compute path | New design and memory agreement | Memory traffic/footprint dominates or token divergence |
| 5 | Newer MLX/MLX-LM version | Needs environment change/install | Fresh pinned baseline and separate version comparison | Gains vanish under matched request semantics |
| 5 | BaseRT, llama.cpp, or another runtime | Needs runtime installation and likely conversion | Separate cross-runtime study, storage agreement first | Non-equivalent quantization/output or no request benefit |
| 6 | Small-row fused quantized matmul | Source proposals available; needs custom kernel/integration | Batch or speculative-verification study, not concurrency-one packing | Microbenchmark win fails dependent request benchmark |
| 6 | Compressed-domain attention/transformed KV | Requires implementation/calibration and quality checks | Long-context research, outside exact-output packing scope | Required-history/quality failure or attention is too small a request fraction |
| 6 | TensorOps/Metal 4 prototype | Requires SDK/device feasibility check and custom code | Separate hardware-specific kernel study | API availability without useful M2 hardware acceleration |

The register is intentionally bounded, not a claim to exhaust every possible algorithm. No new candidate should be added after observing the screen without a new declared question, controls, and decision rule.

## Measurement and adversarial acceptance

Use new matched controls; preserve the forced-output screen and natural-EOS full-workload identities. Require exact token IDs, lengths, EOS/finish behavior, cache offsets/shapes, and required facts/formats before considering latency. Reject rather than loosen a gate after seeing performance. Use clean fresh processes and the same synchronized completion boundary. Preserve individual captures and reversed-order confirmation; report per-fixture and per-length results so pooled medians cannot hide losses. Measure load, packing, first request, request peak, process lifetime RSS, and swap with their distinct meanings. Clean timing must exclude profiling overhead.

The existing two-run order reversal reduces some drift but is not equivalent to a randomized paired-block study. Five screen repetitions and ten per-fixture confirmation repetitions give coarse tails. Keep those preregistered limits; an ambiguous tail warrants an inconclusive result and a new declared confirmation, not repeated runs until passing. No confidence interval, speedup, kernel count, or memory result has been produced by this research review.

## Audit scope and remaining uncertainty

Reviewed: existing Experiment 8 plan/spec, installed Llama and quantized-layer Python source, installed core type signature, pinned upstream Metal source, official release notes, primary research papers, and first-party/contributor-owned source proposals linked above. No MLX import, GPU allocation, timing run, checkpoint conversion, download, dependency installation, or Instruments capture was performed. No model weight bytes were loaded or validated by this review.

The exact executable Metal kernel chosen by the installed wheel, output layout behavior, retained allocation graph, numerical differences, request latency, thermal sensitivity, and peak memory all remain empirical questions. Papers and contributor PRs are leads with different hardware/workloads and do not substitute for local reproduction. Execution must freeze a manifest and verify the installed artifacts rather than assume web documentation describes this environment. An unsuccessful arm is evidence about the declared mechanism/workload, not proof that fusion or other runtimes can never help.

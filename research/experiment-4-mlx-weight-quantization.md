# Experiment 4 research: MLX weight-quantization choices on M2 Pro

Date: 2026-10-03

Target: Llama 3.2 1B Instruct on Apple M2 Pro (16 GB unified memory)

Pinned runtime: MLX 0.32.2, MLX-LM 0.31.3

## Executive conclusion

Experiment 4 should be a **post-training, weight-only quantization comparison**. It changes the stored representation of the already-trained weights and the quantized matrix-multiplication path used during inference. It does not fine-tune the model, change its architecture, or train new weights.

All eight modes or recipes exposed by the pinned `mlx_lm.convert` CLI are syntactically available. For this dense Llama model, their layer shapes are compatible with the required group sizes. The most informative first full-test set is:

1. the current affine 4-bit/group-64 checkpoint as the deployment baseline;
2. a fresh affine 4-bit/group-64 conversion made with the pinned converter as the version-control baseline;
3. BF16 from the same source revision as the quality reference;
4. `mxfp4` and `nvfp4` as same-nominal-bit alternatives; and
5. `mixed_3_6` as a compression-oriented candidate with explicit protection for sensitive layers.

`mxfp8`, `mixed_2_6`, `mixed_3_4`, and `mixed_4_6` are feasible second-wave candidates. MXFP8 roughly doubles the weight payload relative to a 4-bit format, mixed 2/6 is the highest quality-risk option, mixed 3/4 tests whether 4-bit protection is enough, and mixed 4/6 tests whether additional quality is worth a larger model. They should first pass a short compatibility and quality probe, then enter the full suite if they add a useful Pareto point.

The critical conversion rule is: **create every candidate from the same BF16 checkpoint, not from the existing 4-bit model**. Directly invoking quantization on the existing checkpoint does not provide a clean 4-bit-to-new-format transformation, and dequantizing the 4-bit values before requantizing would preserve the old rounding error and bias the comparison.

## What the current checkpoint is

The checked-in local metadata identifies the current model as:

- affine quantization (the omitted `mode` defaults to `affine`);
- 4 bits per stored weight;
- group size 64;
- converted from `mlx-community/Llama-3.2-1B-Instruct-bf16`; and
- approximately 680 MB on disk locally.

That makes the current model a good **deployment baseline**, but the named BF16 parent is the correct common source for creating Experiment 4 variants.

## What MLX-LM 0.31.3 exposes

The pinned converter accepts four uniform modes: `affine`, `mxfp4`, `nvfp4`, and `mxfp8`. It also exposes four mixed-bit recipes: `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, and `mixed_4_6`. A mixed recipe is accepted only with `q_mode=affine`; the converter explicitly rejects combining a mixed predicate with the floating-point modes. The converter and its defaults are visible in the [versioned MLX-LM 0.31.3 source](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/convert.py#L20-L80) and [quantization helper](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/utils.py#L774-L850).

MLX's quantization API defines the following supported combinations ([MLX quantize documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)):

| Choice | Representation | Required/default group | Bit width | Scale/bias representation | Experiment role |
|---|---|---:|---:|---|---|
| affine | group-wise integer-like codes | 64 default; 32/64/128 supported | 2, 3, 4, 5, 6, or 8 | scale and bias, same floating type as input | Current baseline and flexible control |
| mxfp4 | OCP E2M1 microscaled float | 32 | 4 | E8M0 scale, no bias | Same nominal bits; different error and kernel behavior |
| nvfp4 | E2M1 block float | 16 | 4 | E4M3 scale, no bias | Finer groups than MXFP4; potentially different quality/cost |
| mxfp8 | OCP FP8 microscaled float | 32 | 8 | E8M0 scale, no bias | Higher-quality, higher-memory quantized reference |
| mixed_2_6 | affine per layer | 64 by default | mostly 2, selected layers 6 | affine scale and bias | Maximum compression / highest quality risk |
| mixed_3_4 | affine per layer | 64 by default | mostly 3, selected layers 4 | affine scale and bias | Compression-oriented candidate |
| mixed_3_6 | affine per layer | 64 by default | mostly 3, selected layers 6 | affine scale and bias | Quality-protected 3-bit candidate |
| mixed_4_6 | affine per layer | 64 by default | mostly 4, selected layers 6 | affine scale and bias | Quality-oriented candidate over uniform 4-bit |

The OCP specification defines MXFP4 as 4-bit E2M1 values in blocks of 32 with an 8-bit E8M0 shared scale; it defines MXFP8 similarly with 8-bit FP values and block size 32 ([OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)). NVIDIA's native NVFP4 format uses E2M1 values with a block size of 16 and E4M3 block scales plus a tensor-global scale ([NVIDIA NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)). The pinned MLX-LM converter does not supply that optional tensor-global scale. Therefore, treat its `nvfp4` output as MLX's documented group-scaled mode, not as proof of exact interoperability with a complete NVIDIA Blackwell NVFP4 checkpoint. On this Mac it is executed by MLX's Metal backend rather than NVIDIA hardware.

### What the mixed recipes actually protect

The pinned recipe implementation uses higher precision for:

- `v_proj` (and similarly named value-projection layers) in selected transformer blocks;
- `down_proj` in those same selected blocks; and
- `lm_head` in every case.

It selects the first one-eighth and last one-eighth of transformer blocks, plus every third block through the middle. All other eligible modules use the recipe's low bit width. This is a fixed heuristic inspired by mixed quantization choices in llama.cpp, not calibration against this particular Llama checkpoint. That is why measured quality is essential rather than assuming a recipe name is better.

For Llama 3.2 1B's 16 layers, the higher-bit block predicate selects block indices 0, 1, 4, 7, 10, 13, 14, and 15 for the protected `v_proj` and `down_proj` modules. Other eligible projections in those blocks still use the low bit width.

## Runtime and M2 Pro compatibility

MLX 0.32.2 supports these formats through its quantization API, and MLX's `QuantizedLinear` calls `mx.quantized_matmul` directly with the layer's mode, group size, scales, and optional biases ([MLX quantized layer source](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/quantized.py)). MLX 0.30.3 explicitly added NVFP4 and MXFP8 quantized operations to the Metal backend, so those paths are included in the pinned 0.32.2 runtime ([MLX 0.30.3 release](https://github.com/ml-explore/mlx/releases/tag/v0.30.3)).

For this single-device dense Llama workload:

- **Affine modes:** expected to use MLX's Metal quantized matvec/matmul kernels.
- **MXFP4/NVFP4/MXFP8:** expected to use MLX's floating-point quantized Metal kernels.
- **Decode:** one-token generation is a small-row, matrix-vector-shaped workload, so the quantized matvec path is the important one.
- **Prefill:** many prompt tokens make matrix-matrix-shaped work important, so a different dispatch path can win or lose.

Those expected paths are an inference from MLX's public `QuantizedLinear` call and the versioned Metal dispatch/kernel sources: [host-side quantized dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp), [affine Metal kernels](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/quantized.h), and [FP-quantized Metal kernels](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/fp_quantized.h). They should be confirmed with an on-device smoke test because API support does not guarantee that every mode is faster for Llama 3.2 1B's exact shapes.

In particular, the `mxfp*` and `nvfp4` labels do **not** imply native NVIDIA Tensor Core execution on an M2 Pro. MLX provides Metal implementations. Therefore, smaller storage may improve memory traffic, but unpacking/scaling overhead can offset it. The result must be measured separately for prefill and decode.

Activation quantization is a separate experiment. MLX supports `quantize_input=True` only for NVFP4 and MXFP8 linear layers, while the converter comparison here is weight-only. Activation quantization must stay disabled so that the quantized weight format is the only changed variable ([MLX quantized layer source](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/quantized.py)).

The model's quantizable last dimensions (notably 2,048 and 8,192) are divisible by 16, 32, and 64, so the relevant Llama linear/embedding matrices satisfy MLX's group-divisibility rule. The converter also skips any module whose last dimension is not divisible by the requested group size rather than forcing an invalid representation.

## Correct conversion provenance

### Recommended

Use the exact same local BF16 checkpoint and immutable revision for every output. The current model README names `mlx-community/Llama-3.2-1B-Instruct-bf16` as its parent, so that is the natural source. On 2026-10-03, its `main` branch resolved to commit **`863c846a9ac6fad4e49e1743d52984dff262e953`**; pin that full revision rather than downloading mutable `main` ([immutable Hugging Face tree](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-bf16/tree/863c846a9ac6fad4e49e1743d52984dff262e953)). The BF16 model card says it was converted from `meta-llama/Llama-3.2-1B-Instruct` using MLX-LM 0.17.1.

The current 4-bit README says the published checkpoint was produced from that BF16 repository using MLX-LM 0.21.5, while this project pins MLX-LM 0.31.3. Create a fresh affine-4/group-64 model from the pinned BF16 source to isolate converter-version effects, while keeping the published model as the real deployment baseline. Store the BF16 source and every generated candidate under the ignored local `models/` tree.

Record for every conversion:

- source repository and revision/commit;
- SHA-256 of source weight files;
- exact converter command;
- MLX and MLX-LM versions;
- resulting `config.json` quantization block;
- output weight-file hashes and serialized byte count; and
- conversion wall time, as metadata rather than inference performance.

### Why not quantize the existing 4-bit model directly

MLX-LM loads a quantized checkpoint as `QuantizedLinear`/`QuantizedEmbedding` modules. Its `quantize_model` wrapper only selects modules exposing `to_quantized`; already-quantized modules do not expose the same conversion hook. Consequently, pointing `--quantize` at the existing model is not a trustworthy way to transform all layers into another format.

The CLI supports `--dequantize`, so a two-step 4-bit → BF16-like → new-format conversion is technically possible. It is unsuitable for this experiment: dequantization reconstructs approximations of the original weights, not the original BF16 values. A second quantizer would then measure both the old affine error and the new format's error. That unfairly disadvantages every candidate and prevents a clean quality conclusion.

## Fair benchmark design

### Constant conditions

Hold these fixed across all candidates:

- tokenizer and chat template;
- input token IDs, not merely visually identical prompt text;
- 2,048-token prefill step from Experiment 1;
- KV-prefix reuse policy from Experiment 2 when measuring the optimized chat path;
- greedy decoding and the same 128-token output cap;
- prompt suite, conversation histories, stop tokens, and random seed;
- charger/power mode and background-load policy; and
- MLX 0.32.2 / MLX-LM 0.31.3.

Use a fresh process for cold-load measurements. For steady-state inference, load once, run at least one warmup, then use the same repeated-run count for every candidate. Randomize or rotate candidate order between rounds so temperature and background drift do not always favor one model.

### Performance and footprint measurements

Measure both prefill and decode because the kernel shapes differ:

- model load time and first-call latency, reported separately;
- prompt/prefill tokens per second at the existing short and long prompt lengths;
- time to first token;
- steady decode tokens per second;
- end-to-end request/session time;
- MLX peak and active memory;
- process RSS or physical footprint;
- serialized model bytes; and
- optional GPU energy/time only if both arms receive matching telemetry captures.

Report per-prompt results plus median and p95 across repetitions. Do not claim an energy or thermal improvement from an unmatched recording.

### Quality measurements

Exact output equality is a useful diagnostic but should not be the only acceptance rule: quantization is expected to perturb logits, especially around near-ties. Use three layers of evidence:

1. **Teacher-forced loss/perplexity:** calculate NLL/perplexity on one fixed, versioned text corpus or local fixture set with the same tokenizer. BF16 is the quality reference.
2. **Logit stability:** on fixed token positions, record top-1 agreement and a distribution-distance measure such as KL divergence against BF16.
3. **Task behavior:** reuse fixed Q&A, extraction, rewrite, summarization, and multi-turn fixtures; check required facts, contradictions, instruction following, and usefulness.

Do not choose a universal quality threshold silently. The acceptable perplexity delta, task-failure allowance, minimum speed gain, and memory target are product choices and should be confirmed before the full sweep. The experiment should publish the whole trade-off table even if no candidate satisfies every desired threshold.

### Staged execution

1. **Source gate:** obtain and hash one BF16 source revision.
2. **Conversion gate:** convert each mode to a unique output directory; verify config, loadability, one forward pass, and one short generation.
3. **Quality probe:** run a small NLL/logit/task subset. Reject crashes, NaNs, incoherent output, or severe loss regressions early.
4. **Full benchmark:** run the existing fixed workload on the high-value survivors.
5. **Combined confirmation:** test the winner with prefill step 2,048 and cross-turn KV-prefix reuse enabled.
6. **Decision:** select a Pareto winner rather than assuming fewer bits is automatically better.

## Candidate priority and expected value

| Candidate | Feasible? | Full-suite priority | Why |
|---|---|---|---|
| published affine 4-bit / group 64 | Yes; already running | Required deployment baseline | Current behavior and strongest existing evidence; produced with MLX-LM 0.21.5 |
| fresh affine 4-bit / group 64 | Yes | Required version-control baseline | Separates quantization-format effects from converter-version effects |
| BF16 | Yes | Required quality reference | Clean reference for loss/logits; memory/performance ceiling |
| mxfp4 | Yes, pending on-device smoke | High | Same nominal 4 bits with different block-float error and Metal kernel path |
| nvfp4 | Yes, pending on-device smoke | High | Same nominal 4 bits with 16-value blocks; may improve error but adds scale overhead |
| mixed_3_6 | Yes | High | Meaningful compression with stronger protection for selected layers |
| mixed_3_4 | Yes | Medium | Tests whether lighter 4-bit protection is sufficient |
| mixed_4_6 | Yes | Medium | Tests whether modest extra precision buys quality at acceptable cost |
| mixed_2_6 | Yes | Low / probe first | Maximum compression but highest probability of a material quality loss |
| mxfp8 | Yes, pending on-device smoke | Low / reference | More weight traffic than 4-bit modes; likely quality-focused rather than a speed/size winner on this 1B model |

## Decision guidance

The most likely outcomes worth learning from are:

- **Affine 4-bit remains best:** the current format already matches the M2 Pro's practical speed/quality balance.
- **MXFP4 or NVFP4 wins:** same nominal weight width but a better kernel/error trade-off on this runtime.
- **Mixed 3/6 wins:** a smaller model with acceptable quality and no harmful decode slowdown.
- **Mixed 4/6 wins:** slightly larger weights but measurably better quality with similar latency.

A candidate should not be kept merely because its model file is smaller. The useful result is the best combined point across answer quality, TTFT, prefill throughput, decode throughput, and peak memory. If every alternative loses to the current affine 4-bit checkpoint on that combined view, rejecting the Experiment 4 change is the correct result.

## Primary sources

- [MLX-LM 0.31.3 converter source](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/convert.py)
- [MLX-LM 0.31.3 quantization/load utilities](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/utils.py)
- [MLX `quantize` API and supported-mode table](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- [MLX `quantized_matmul` API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html)
- [MLX 0.32.2 quantized neural-network layers](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/quantized.py)
- [MLX 0.32.2 Metal quantized dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp)
- [MLX 0.32.2 affine quantized Metal kernels](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/quantized.h)
- [MLX 0.32.2 FP-quantized Metal kernels](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/fp_quantized.h)
- [MLX 0.30.3 release: NVFP4/MXFP8 Metal support](https://github.com/ml-explore/mlx/releases/tag/v0.30.3)
- [OCP Microscaling Formats specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA NVFP4 format documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
- [Pinned BF16 source revision on Hugging Face](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-bf16/tree/863c846a9ac6fad4e49e1743d52984dff262e953)

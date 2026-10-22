# Apple Silicon inference engineering for Llama 3.2 1B with MLX-LM

This repository studies inference engineering for Llama 3.2 1B Instruct using MLX and MLX-LM on Apple Silicon, primarily an M2 Pro with 16 GB unified memory.
Across eight experiments, it configures and evaluates MLX-LM's existing prefill, KV-cache reuse, quantization, and speculative-decoding features, and tests experiment-local compiled-decode and packed-projection variants.
The project contributes benchmark and evaluation harnesses, integration policies, controlled comparisons, correctness checks, and profiling evidence built on the upstream runtime.
The results favor retaining published affine 4-bit/group-64 weights and a 2,048-token prefill step, with existing exact-prefix cache reuse reducing median paired cache-hit TTFT by 71.49% versus fresh-cache recomputation in its clean sequential-chat confirmation.
The other tested candidates did not meet their adoption gates, and GPU profiling identifies quantized matrix work as the next research target without establishing a custom-kernel speedup.

## Scope and measurement

The primary control is `mlx-community/Llama-3.2-1B-Instruct-4bit`, with
MLX 0.32.2 and MLX-LM 0.31.3. **The study covers one model family and parameter
size: Llama 3.2 1B Instruct, evaluated in multiple precision/quantization
artifacts.** Experiment 4 includes a BF16 reference, a fresh affine-4 control,
MXFP4, NVFP4, and Mixed-3/6 variants; Experiment 6 uses the same full-1B
architecture in Mixed-3/6 form as its draft. It is not a comparison across
different model families or sizes. The [artifact manifest](benchmarks/results/experiment4-artifact-manifest-20261003.json)
records the distinct source revision, conversion recipes, and artifact hashes;
published and freshly converted weights are separate controls.

No training or fine-tuning is performed. These are benchmark-backed runtime
selections, not a production chat server or an integrated deployment.

**Prefill** processes the input prompt; **decode** generates subsequent tokens.
**TTFT** means time to first token. Most tests use greedy decoding, concurrency
one, and a fresh request cache unless reuse is the treatment. Each experiment
has its own workload and advancement rules: a failed bounded screen is not a
full confirmation, and a correct output is not automatically a useful speedup.
Clean timings are kept separate from profiling, except for the explicitly
instrumented Experiment 1 sweep. Memory figures distinguish model/cache bytes,
MLX active or peak allocations, and process memory.

Results apply to the tested model, runtime, hardware, and workloads. OS and
power conditions changed across dates, so timings from different experiments
are not a controlled before/after comparison. M4 Pro evidence is a separate
hardware dataset. [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) records the detailed
decisions, limitations, and experiment history.

## Upstream features and project contributions

Most experiments evaluate mechanisms already implemented by MLX/MLX-LM.
They ask when and how to use those mechanisms for this workload, rather than
claiming to invent them or outperform an equivalently configured upstream
runtime. In particular, the cache-reuse improvement is **reuse versus fresh
recomputation**, not a faster replacement for MLX-LM's cache implementation.

| Experiment | Existing runtime capability | What this repository tests or adds |
|---|---|---|
| 1: prefill | MLX-LM `prefill_step_size` | A controlled chunk-size sweep with latency/memory evidence |
| 2: prefix reuse | MLX-LM `LRUPromptCache`, `make_prompt_cache`, and cached generation | A multi-turn fresh-versus-reuse comparison, session lifecycle checks, and output/quality validation |
| 3: compiled decode | MLX `mx.compile` and MLX-LM cache/model objects | An explicit state-in/state-out compilation boundary and bounded retracing/correctness probes |
| 4: weight formats | MLX-LM conversion and MLX quantization formats | Pinned-source artifact comparisons and quality/performance selection gates |
| 5: quantized KV | MLX-LM `QuantizedKVCache`, bit/group settings, and `quantized_kv_start` | Immediate versus prompt-boundary conversion policies, cache-byte accounting, and quality/memory checks |
| 6: speculation | MLX-LM `stream_generate` with `draft_model` and `num_draft_tokens` | A particular target/draft pairing, draft-count sweep, zero-draft control, and acceptance/correctness measurements |
| 7: profiling | MLX measurements and Apple profiling tools | Matched request-phase captures, synchronized phase attribution, and isolated KV-update diagnostics |
| 8: packing | MLX quantized matrix primitives and MLX-LM Llama modules | An experiment-local row-packing adapter for Q/K/V and gate/up projections, with unchanged parameter bytes and numerical/token/cache gates |

Experiments 3 and 8 change the execution arrangement beyond choosing a runtime
option; they still use existing MLX primitives and GPU kernels. No custom
Metal kernel or independent inference runtime was implemented in V1. The
reusable output is the measurement procedure, tested harnesses, preserved data,
and scoped positive/negative findings; it does not establish a new general
inference algorithm.

## Experiments and findings

### Baseline: establish a repeatable reference

The expanded baseline tests five tasks—question answering, rewriting,
contextual questions, JSON extraction, and summarization—at target prompt
lengths of 64, 512, 2,048, and 4,096 tokens. Each fixture has one warmup and ten
measured repetitions: **200 measured generations**, capped at 128 output
tokens. Actual input lengths, output lengths/text, timing, memory, and runtime
metadata are preserved.

This establishes a diverse reference rather than relying on one favorable
prompt. The earlier single-prompt capture remains historical evidence, but is
not an equivalent workload. These are performance fixtures, not a comprehensive
quality benchmark; very short prompts can omit facts needed to answer a task.

Evidence: [diverse baseline JSON](benchmarks/results/diverse-baseline-20260925T053031Z.json)
and [benchmark protocol](.agents/skills/mlx-inference-benchmark/references/benchmark-protocol.md).

### Experiment 1: prompt-prefill step size

Prefill step size controls how many input tokens MLX-LM processes in a chunk.
We swept **512, 1,024, 2,048, and 4,096**, holding the model and diverse task
suite fixed. Larger chunks can change throughput and temporary-memory demand.

**Keep 2,048; retain 512 as a memory-oriented option.** At 4,096-token inputs,
step 2,048 had median TTFT of **2.161 s**, versus **2.249 s** for step 512;
step 512 used **1.483 GB** peak MLX memory versus **1.635 GB**. Step 4,096 was
slower and showed severe tail outliers whose cause was not established.
The sweep ran in ascending order with `powermetrics` active, so these are
instrumented comparisons, not clean confirmation of a universal optimum.

Evidence: [analysis and all four captures](benchmarks/results/prefill-step-size-analysis.md),
[2,048-step data](benchmarks/results/prefill-step-2048-telemetry-20260925T062407Z.json),
and [512-step data](benchmarks/results/prefill-step-512-telemetry-20260925T060849Z.json).

### Experiment 2: cross-turn KV-prefix reuse

A KV cache stores attention keys and values for processed tokens. Reusing an
**exact cached token prefix** avoids processing the same conversation history
again. The runner uses MLX-LM's existing `LRUPromptCache` and generation
APIs; it does not implement a new attention-cache algorithm. We compared
fresh-cache generation with reuse over five four-turn scenarios, with one
warmup and ten measured sessions per scenario.

**Select reuse as a performance candidate for sequential chat.** Across 150
measured cache-hit turns in the clean confirmation, median paired TTFT
improved **71.49%**; paired session peak MLX memory fell **6.79%**. All required
facts passed, and all 200 measured turns happened to match control tokens.
An earlier capture produced a wording difference without losing the required
answer, so exact-token identity is not a general guarantee. Reuse requires
correct prefix matching, state isolation, cache bounds, and invalidation;
unrelated requests still need fresh caches. No matched power or thermal gain
was measured.

Evidence: [practical analysis and caveats](benchmarks/results/experiment2-practical-analysis.md),
[control](benchmarks/results/experiment2-control-practical-timing-20260926T190000Z.json),
and [reuse confirmation](benchmarks/results/experiment2-reuse-practical-confirmation-20260926T190500Z.json).

### Experiment 3: explicit compiled decode

We wrapped the single-token model/sampling path in `mx.compile`, explicitly
passing KV-cache state into and out of the compiled function. The question was
whether graph compilation could reduce decode overhead while preserving cache
behavior and greedy tokens.

**Reject this compilation boundary on the pinned runtime.** Shapeless
compilation failed to infer dynamic cache-slice output shapes. A bounded
shape-specific fallback matched all eight tokens but **retraced once per
token**: first-token latency was about **17× higher**, subsequent decode was
**36.9% slower**, and peak MLX memory rose **10.6%**. The feasibility gate
stopped the full suite. This rejects the tested dynamic-cache boundary, not
compilation generally; fixed-shape state or a smaller stateless subgraph would
be a different experiment.

Evidence: [analysis](benchmarks/results/experiment3-compiled-decode-analysis.md),
[shapeless failure](benchmarks/results/experiment3-compile-probe-20261003T-stage-a-retry1.json),
and [eight-token fallback](benchmarks/results/experiment3-compile-probe-20261003T-shaped-8.json).

### Experiment 4: weight quantization formats

Weight quantization changes how model parameters are encoded, trading storage,
compute, and numerical behavior. We compared the published affine-4/group-64
model with fresh affine-4, MXFP4, NVFP4, and Mixed-3/6 artifacts converted from
the same pinned BF16 source. The adoption rule required at least **10% faster
generation**, or **20% lower model size and peak memory**, while preserving
practical quality and TTFT.

**Keep the published affine-4 weights.** MXFP4 and Mixed-3/6 lost baseline
quality capabilities and saved only **5.6%** and **14.5%** in weight bytes.
NVFP4 reached the performance screen, but pooled median decode throughput fell
**5.69%**, with negligible size/memory savings. It also omitted a baseline
rewrite detail despite a higher aggregate quality score. The affine controls
passed only five of ten quality cases themselves, so the comparison concerns
preservation of existing capabilities, not perfect model quality. No candidate
advanced to full confirmation.

Evidence: [analysis](benchmarks/results/experiment4-weight-quantization-analysis.md),
[artifact manifest](benchmarks/results/experiment4-artifact-manifest-20261003.json),
[fresh affine screen](benchmarks/results/experiment4-screen-fresh-affine4-20261003.json),
and [NVFP4 screen](benchmarks/results/experiment4-screen-nvfp4-20261003.json).

### Experiment 5: KV-cache quantization

Unlike weight quantization, this compresses the growing attention cache.
We tested 8-bit and 4-bit caches at 4K, 8K, and 16K contexts, both during
prefill and with conversion postponed until the prompt boundary using
MLX-LM's existing `quantized_kv_start` setting.

**Reject all tested arms under the declared gates.** Logical cache bytes per
token fell about **46.9%** with 8-bit and **71.9%** with 4-bit, but immediate
quantization raised 16K peak MLX memory from **1.991 GB** to **6.120/6.842 GB**.
Post-prefill conversion avoided that spike but reduced whole-request peak
memory by **0%**, missing the required 20% reduction. Post-prefill 8-bit also
introduced a new three-bullet-to-two-bullet format failure at 8K; 4-bit failed
quality earlier. Exploratory decode-throughput gains were not confirmed with
repeated identical work.

Independent review corrected the initial 8-bit quality interpretation and
identified a missing history fixture and a permissive JSON evaluator. Those
limitations remain explicit; the original raw results are preserved.

Evidence: [corrected analysis](benchmarks/results/experiment5-kv-cache-quantization-analysis.md),
[review corrections](benchmarks/results/experiment5-review-corrections.md),
[16K control](benchmarks/results/experiment5-smoke-v1-16k-control-20261003.json),
[post-prefill 8-bit smoke](benchmarks/results/experiment5-smoke-postprefill-16k-8bit-20261003.json),
and [8K/16K quality data](benchmarks/results/experiment5-quality-postprefill-8bit-8k-16k-20261003.json).

### Experiment 6: speculative decoding

Speculative decoding uses a draft model to propose tokens that the target
verifies in batches. We used MLX-LM's built-in speculative-generation path
and paired the affine-4 target with the Mixed-3/6 draft,
testing one to four draft tokens plus a zero-draft control. The draft retained
the target's full 16-layer, 1B architecture; its weight file was only 14.5%
smaller.

**Reject this target/draft pairing.** All four draft counts passed the
seven-case exact-token gate, but every arm was slower. Even the best real
draft arm increased pooled median forced-128-token request time by **112.04%**
and reduced decode throughput **58.67%**. Keeping both models resident raised
post-load active MLX memory **85.5%**, before request caches. The zero-draft
path alone increased request time **65.29%**. Useful proposal acceptance could
not offset the cost of a second full-sized model; no candidate advanced past
the bounded screen. This does not reject cheaper drafts or self-speculation.

Evidence: [analysis](benchmarks/results/experiment6-speculative-decoding-analysis.md),
[correctness gate](benchmarks/results/experiment6-stage0-mixed3_6-20261004.json),
[target control](benchmarks/results/experiment6-stage1-target-20261004.json),
and [one-token draft screen](benchmarks/results/experiment6-stage1-draft1-20261004.json).

### Experiment 7: runtime and GPU profiling

This diagnostic asks **where time is spent**, rather than testing a faster
implementation. Clean forced-128-token runs separate TTFT from subsequent
decode; separately synchronized Metal recordings rank GPU operation families
and isolate the actual KV-cache update code with synthetic inputs at model
geometry.

**Investigate quantized matrix work first; no custom-kernel win is established.**
In the clean M2 confirmation, decode accounted for a median **82.5%** of
64-token request time, while TTFT accounted for **74.0%** at 4,096 tokens.
In separately bounded M2 recordings, quantized matrix/dequantization work
contributed **91.53% of prefill** and **88.49% of decode summed shader sample
duration**. These are instrumented sample shares, not clean request fractions
or predicted speedups. Isolated KV append, allocation, and growth costs were
also measured; those synthetic timings cannot establish the cache's fraction
of an ordinary request.

The diagnostic completed with validated CLI evidence. The original M2 GPU
captures remain unreplayed. Separate M4 Pro/24 GB replay evidence also ranked
quantized matrix operations highest, but its hardware and shader-cost metric
must not be pooled with the M2 data.

Evidence: [clean M2 confirmation](benchmarks/results/experiment7-m2pro-confirmation-analysis-20261019.md),
[completed phase/KV analysis](benchmarks/results/experiment7-m2pro-phase-and-kv-analysis-20261020.md),
[prefill attribution](benchmarks/results/experiment7-m2pro-phase-prefill-20261020-r2-attribution.json),
[decode attribution](benchmarks/results/experiment7-m2pro-phase-decode-20261020-r2-attribution.json),
[validation](benchmarks/results/experiment7-m2pro-phase-validation-20261020.json),
and [separate M4 report](benchmarks/results/experiment7-m4pro-runtime-profile-analysis.md).

### Experiment 8: packed quantized projections

Some transformer projections share an input. We packed their quantized weight
rows so one larger projection could replace separate calls: **Q** combines
query/key/value, **G** combines gate/up, and **QG** combines both. The test
preserved published parameter bytes and required numerical, exact-token/cache,
speed, and memory gates before adoption.

**Reject all three packing arms.** All **13,440 component numerical
comparisons** passed the declared tolerances, but Q and QG changed the first
greedy token on one short context question and stopped before timing. A
separate diagnostic reproduced sensitivity near an exposed fp16 log-probability
tie. Numerical closeness alone did not preserve exact tokens.

G matched all **65 single-request/history/cache cases**, but short-request
median time changed only **-1.07%** at target 64 and **+2.17%** at target 512,
missing the 5% benefit screen. Request peak MLX memory rose **10.01–12.18%**
at the three longer lengths, exceeding the 5% allowance. The screen used five
measured repetitions per length and arm; no finalist qualified for the full
suite or reversed confirmation. Fewer projection calls did not establish fewer
GPU dispatches or less memory traffic.

Control-quality limitations include inadequate notes in a short fixture,
weekday/bullet validator false negatives, and missing extraction schemas.
Rejected setup attempts and those limitations remain disclosed; they do not
reverse the failed adoption gates.

Evidence: [analysis](benchmarks/results/experiment8-packed-projections-analysis.md),
[plan](experiments/experiment-8-packed-projections-plan.md),
[specification](experiments/experiment-8-packed-projections-spec.md),
[G correctness data](benchmarks/results/experiment8-packed-G-correctness-20261020-r3.json),
[control screen](benchmarks/results/experiment8-packed-C-screen-20261020-r3.json),
[G screen](benchmarks/results/experiment8-packed-G-screen-20261020-r3.json),
and [recomputed comparisons](benchmarks/results/experiment8-packed-screen-analysis-20261020-final.json).

## Next phase

V1's eight experiments are complete at their documented decision gates.
The next phase is a separate V2 Metal-kernel study, starting with **R0:
evaluation-contract repair**, then a matched stock baseline and measured
kernel selection. R0 belongs to V2 and does not change V1's recorded results;
no custom Metal kernel is implemented in this repository.

## Repository map

- `benchmarks/`: runners, deterministic fixtures, raw JSON results, and analyses.
- `experiments/`: experiment plans, specifications, and decision rules.
- `research/`: primary-source research and design notes.
- `.agents/skills/`: reusable benchmark and prefill-tuning procedures.

## Local setup

This is a completed research archive. To reproduce a capture, provide your own
local environment and model checkpoint using the versions, workload and settings
recorded with that experiment. The primary runtime is MLX 0.32.2 and MLX-LM
0.31.3; the baseline uses Python 3.14.4. Treat a different machine or software
version as a new condition requiring its own matched control.

The repository expects, but intentionally does not contain:

- `.venv/`
- `models/Llama-3.2-1B-Instruct-4bit/`

With the existing environment and model available, run from the repository root:

```bash
./.venv/bin/python benchmarks/run_baseline.py \
  --output benchmarks/results/<unique-run-id>.json
```

Never overwrite a prior result. Keep primary timing runs separate from
`powermetrics` or Metal-capture diagnostics. Use the recorded experiment
protocol for comparisons; a new OS, runtime, model, or power setting requires
a matched control.

## Data boundaries

Model artifacts, virtual environments, the private research vault, and raw
`.gputrace`/`.trace` recordings stay local and ignored. Raw GPU traces can be
large and contain machine-local metadata. Committed telemetry logs have
process-name and PID tables removed. Selected Experiment 8 JSON captures,
including excluded attempts and rejected-arm correctness results, are retained
as gzip archives with verified recovery and
provenance. Their analyses distinguish accepted gate evidence from excluded
execution attempts.

## Publication provenance

The public study is hosted at
[mlx-llama-3-2-1b-inference-lab](https://github.com/sabcodes/mlx-llama-3-2-1b-inference-lab).
Before publication, eight metadata instances were removed from the research
history: a local model path, four system boot-time values, and three timezone
annotations. Executable code, generated outputs, numerical measurements and
compressed evidence are unchanged. Historical records can retain their original
commit labels; the [commit map](research/publication-history-map-20261022.json)
resolves those labels to the cleaned revisions in this repository. This migration
does not represent a rerun or a new experimental result.

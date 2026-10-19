# Experiment 7 plan: profile the MLX-LM runtime

Date: 2026-10-04

## Question

Where does this Mac spend inference time for Llama 3.2 1B 4-bit, and is there
enough concentrated cost to justify a custom MLX or Metal optimization?

This is a diagnostic experiment. It does not change the model or deployed
runtime setting.

## Why this comes now

Experiments 1–6 tested exposed runtime and representation choices. Some helped
specific workloads and several were correctly rejected. Before writing custom
kernels, we need evidence that a particular phase or operation is both
dominant and insufficiently optimized by MLX.

## Controlled timing profile

- Same local Llama 3.2 1B Instruct affine-4/group-64 weights.
- MLX 0.32.2 and MLX-LM 0.31.3 from the project `.venv`.
- Greedy generation, one request at a time, fresh KV cache per request.
- Selected 2,048-token prefill step size from Experiment 1.
- Deterministic summarization fixture at approximately 64, 512, 2,048, and
  4,096 prompt tokens.
- EOS suppressed so every request generates exactly 128 tokens.
- One warmup and ten measured runs per input length.
- No `powermetrics` during primary timing.

The runner records request time, time to first token (TTFT), time after the
first token, measured decode throughput, median/p95 inter-token latency, MLX
memory, and process peak RSS. TTFT is deliberately named rather than called
"pure prefill": it also contains the first token's model work and sampling.

## Focused Metal captures

The installed MLX exposes `mx.metal.start_capture()` and `stop_capture()`. Two
small optional `.gputrace` captures isolate approximately:

1. prompt processing plus first-token work; and
2. a steady sequence of decode steps after prompt processing.

MLX generation pipelines one token ahead, so capture boundaries are
approximate and the metadata says so. Raw traces are gitignored: they can be
large and may contain local machine metadata. The Mac currently has Command
Line Tools rather than full Xcode, so kernel-level inspection must wait until
Xcode's Metal Debugger is available.

MLX requires the capture process to start with `MTL_CAPTURE_ENABLED=1`. The
profile script warms normal kernels before recording to keep one-time startup
compilation out of the trace.

## Decision rules

This experiment succeeds when it identifies the dominant request phase and
creates reproducible evidence for or against deeper kernel work.

- A phase is dominant when it occupies at least 50% of median end-to-end
  request time for a workload.
- A custom-kernel follow-up is warranted only if a Metal trace identifies an
  operation family whose end-to-end share leaves a plausible path to at least
  a 10% request-time improvement.
- A microbenchmark improvement by itself is not enough. Any later prototype
  must preserve tokens/quality and improve the full benchmark.

If existing MLX quantized-matmul or attention kernels dominate without an
obvious redundant or unfused sequence, the correct result is to avoid a custom
kernel and choose a higher-level optimization target.

## Commands

Primary clean timing:

```bash
.venv/bin/python benchmarks/run_experiment7_profile.py \
  --mode timing \
  --output benchmarks/results/experiment7-phase-profile-20261004.json
```

Optional focused captures (raw traces remain local):

```bash
MTL_CAPTURE_ENABLED=1 .venv/bin/python benchmarks/run_experiment7_profile.py \
  --mode capture --capture-phase prefill \
  --trace-output benchmarks/results/experiment7-prefill-20261004.gputrace \
  --output benchmarks/results/experiment7-prefill-capture-20261004.json

MTL_CAPTURE_ENABLED=1 .venv/bin/python benchmarks/run_experiment7_profile.py \
  --mode capture --capture-phase decode \
  --trace-output benchmarks/results/experiment7-decode-20261004.gputrace \
  --output benchmarks/results/experiment7-decode-capture-20261004.json
```

## Primary references

- [MLX Metal Debugger documentation](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)
- [MLX Metal Python API](https://ml-explore.github.io/mlx/build/html/python/metal.html)
- [Apple: Capturing a Metal workload programmatically](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically)
- [Apple Metal tools overview](https://developer.apple.com/metal/)

## Completion follow-up, 2026-10-19

The [completion specification](../experiments/experiment-7-runtime-profiling-spec.md)
and [execution plan](../experiments/experiment-7-runtime-profiling-plan.md)
record the user-directed review of Arno's M4 work. See the
[review](../benchmarks/results/experiment7-review-20261019.md) for validated phase
results, reported replay evidence, provenance limits, and Experiment 8 status.

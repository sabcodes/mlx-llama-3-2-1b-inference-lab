# Experiment 4 start notes

Date: 2026-10-03

## What has started

Experiment 4 is comparing post-training weight representations for the same
Llama 3.2 1B Instruct model. The pinned BF16 source and all converted model
directories are under ignored `models/`; model weights are not committed.

The common source is
`mlx-community/Llama-3.2-1B-Instruct-bf16` at revision
`863c846a9ac6fad4e49e1743d52984dff262e953`. Artifact hashes, byte counts, and
reproduction commands are in
`benchmarks/results/experiment4-artifact-manifest-20261003.json`.

## Fresh deployment-control baseline

The existing published affine-4/group-64 model completed the full fixed suite:
five prompt types at four input lengths, with one warmup and ten measured runs
per prompt. There were 200 measured runs and no empty outputs.

| Input target | Median TTFT (s) | Median prefill tok/s | Median decode tok/s | Median MLX peak GB |
|---:|---:|---:|---:|---:|
| 64 | 0.1459 | 1,146.4 | 224.6 | 0.833 |
| 512 | 0.3967 | 1,670.9 | 197.2 | 1.230 |
| 2,048 | 1.2562 | 1,758.5 | 170.3 | 1.542 |
| 4,096 | 2.6457 | 1,601.5 | 138.4 | 1.635 |

These values are the Experiment 4 deployment-control baseline, not a candidate
comparison yet.

## Source and conversion gates

The BF16 source loaded and generated successfully. The fresh affine-4 control,
MXFP4, NVFP4, and mixed-3/6 artifacts all converted, loaded, and generated on
the M2 Pro with the pinned runtime.

| Artifact | Serialized weights | Relative to published affine-4 |
|---|---:|---:|
| BF16 source | 2,471,645,521 B | +255.5% |
| published affine-4/g64 | 695,283,921 B | control |
| fresh affine-4/g64 | 695,284,180 B | approximately equal |
| MXFP4/g32 | 656,654,304 B | -5.6% |
| NVFP4/g16 | 695,271,493 B | approximately equal |
| mixed-3/6/g64 | 594,293,188 B | -14.5% |

## Initial quality gate

The 64-token fixtures remain useful for speed but are too short to consistently
contain all task facts, so candidate quality probes use the 512-token fixtures.

- BF16, fresh affine-4, MXFP4, and NVFP4 answered the direct weekday-closing
  and meeting-room questions correctly.
- `mixed_3_6` incorrectly claimed that the meeting room was not mentioned.
  This loses an explicitly supplied fact, so it is rejected before the full
  benchmark.
- MXFP4's extraction answer omitted the requested weekday closing time.
- NVFP4's summarization added an unsupported claim that the library is open
  seven days a week.
- The BF16 reference itself added unsupported weekend hours in its rewrite and
  summary, showing why “matches BF16 wording” is not a sufficient quality rule.

MXFP4 and NVFP4 therefore remain provisional. Their direct answers work, but
their observed task-level regressions must be resolved by the fixed quality
gate before either is called a survivor. The single-run probe timings are not
used to claim a performance win.

## Next gate

1. Add a deterministic quality evaluator for required facts, requested JSON
   fields, unsupported scheduling claims, and output corruption.
2. Run the quality gate on the published and fresh affine-4 controls plus
   MXFP4 and NVFP4.
3. Run the short performance screen only on models that pass quality.
4. Run the full fixed suite and a confirmation rerun only on candidates that
   show a plausible speed/size benefit.

No Experiment 4 winner has been selected yet.

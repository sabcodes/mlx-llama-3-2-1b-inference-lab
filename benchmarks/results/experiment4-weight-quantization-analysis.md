# Experiment 4 analysis: MLX weight quantization formats

Date: 2026-10-03

## Decision

Keep the existing published affine 4-bit/group-64 model. None of the first-wave
alternatives satisfied the approved decision rule:

- at least 10% faster generation, **or**
- at least 20% lower model size and peak memory,
- while keeping practical answer quality and TTFT roughly as good as the
  current model.

This is a useful negative result: the newer floating-point 4-bit formats are
supported by MLX on this M2 Pro, but support does not make them faster for this
model's matrix shapes.

## Controlled setup

Every new artifact was converted from the same pinned BF16 source:
`mlx-community/Llama-3.2-1B-Instruct-bf16` revision
`863c846a9ac6fad4e49e1743d52984dff262e953`. No candidate was derived from the
existing 4-bit weights. MLX 0.32.2, MLX-LM 0.31.3, greedy decoding, a 128-token
generation cap, prefill step size 2,048, and concurrency one were held fixed.

The full deployment-control capture contains five prompts at each of four
input lengths, one warmup, and ten measured repetitions (200 measured runs).
Candidate performance screening used the same tasks and lengths with three
measured repetitions per prompt (60 measured runs per model). A full candidate
run was reserved for a model showing a plausible route to the win threshold.

## Practical-quality gate

The fixed quality suite contains ten deterministic cases covering supplied
facts, conditional facts, meaning-preserving rewriting, structured extraction,
grounded summarization, negation, two-step lookup, literal instructions,
unknown facts, and long-context key recall.

The current and fresh affine controls both passed the same five cases. Because
this 1B model does not pass every strict instruction even in BF16, candidate
quality was judged relative to the current deployed model rather than against
an unrealistic perfect score.

| Model | Cases passed | Quality result |
|---|---:|---|
| Published affine-4/g64 | 5/10 | Deployment baseline |
| Fresh affine-4/g64 | 5/10 | Same passed-case set as deployment baseline |
| BF16 | 5/10 | Diagnostic reference; different passed-case set |
| MXFP4/g32 | 4/10 | Rejected; lost the baseline's two-step lookup capability |
| NVFP4/g16 | 8/10 | Provisional quality survivor; omitted the receipt detail in one rewrite |
| Mixed-3/6/g64 | 5/10 | Rejected; lost several baseline facts/tasks despite gains elsewhere |

NVFP4's aggregate score was higher, but it did not strictly preserve every
baseline-passed detail. It was allowed into the performance screen because the
remaining quality question could matter only if it delivered a material
performance or memory benefit.

## Artifact size

| Model | Weight bytes | Change vs published affine-4 |
|---|---:|---:|
| Published affine-4/g64 | 695,283,921 | control |
| Fresh affine-4/g64 | 695,284,180 | approximately 0% |
| MXFP4/g32 | 656,654,304 | -5.6% |
| NVFP4/g16 | 695,271,493 | approximately 0% |
| Mixed-3/6/g64 | 594,293,188 | -14.5% |

Neither smaller artifact reached the required 20% size reduction, and both
failed the practical-quality gate before full performance testing.

## NVFP4 performance screen

The fair screen comparison is NVFP4 against the fresh affine-4 model created
from the same BF16 source with the same converter.

| Input tokens | Decode tok/s: affine → NVFP4 | Decode change | TTFT change | Peak MLX memory change |
|---:|---:|---:|---:|---:|
| 64 | 189.40 → 180.62 | -4.64% | +1.73% | +0.42% |
| 512 | 175.62 → 169.18 | -3.67% | +3.55% | +0.98% |
| 2,048 | 145.21 → 139.28 | -4.08% | +4.04% | -1.63% |
| 4,096 | 118.47 → 115.04 | -2.90% | +4.59% | approximately 0% |

Across all 60 measured requests, median decode throughput changed from 169.65
to 160.00 tokens/second (-5.69%), median TTFT changed from 0.9754 to 1.0286
seconds (+5.46%), median prefill throughput fell by 4.94%, and median MLX peak
memory fell by only 0.47%. NVFP4's weight file is only 12,428 bytes smaller
than the fresh affine control.

NVFP4 therefore moves in the wrong direction for speed and is nowhere near the
20% size-and-memory route. A ten-repeat full benchmark cannot change its file
size and would need to reverse a consistent loss at every tested length into a
greater-than-10% speedup. It is stopped at the screening gate.

## Final outcome

- **Selected format:** existing affine 4-bit, group size 64.
- **MXFP4:** rejected for quality and insufficient size reduction.
- **NVFP4:** rejected for speed/size; also carries one detail-preservation caveat.
- **Mixed 3/6:** rejected for practical-quality loss and insufficient size reduction.
- **Full candidate confirmation:** not run because no candidate qualified from
  the predeclared screening funnel.

Experiment 4 does not change the deployed model. The next optimization should
target a different inference mechanism rather than another first-wave weight
format.

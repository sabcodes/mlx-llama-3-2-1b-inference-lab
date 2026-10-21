# Experiment 8: packed quantized projections on local M2

Date: 2026-10-20. **Decision: reject the tested packing
arms; retain the existing runtime configuration.** All declared arms have a
resolved disposition. No eligible finalist remains, so the conditional full
comparison (200 measurements per arm) and reversed confirmation were not run.

This is a bounded negative result for this model/runtime/workload, not a claim
that projection packing, fusion, or other Apple Silicon runtimes are generally
ineffective. No >=10% end-to-end win or production deployment is established.

## Preregistered question and execution

After Experiment 7 identified quantized matrix work as the dominant sampled
shader family, test whether row-packing same-input projections improves short
requests without changing the published affine-4/group-64 parameters or exact
greedy output. The plan was refreshed first, specification second, implementation
third. Research and adversarial review added quantized dispatch boundaries,
precise generation-stream/lookahead timing, cache/power/capacity checks, and
fail-closed output-gate definitions before candidate performance was observed;
the offline analyzer was completed before primary timing capture.

Environment: Apple M2 Pro, 16 GiB unified memory, macOS 27.0.1, Python 3.14.4,
MLX 0.32.2, MLX-LM 0.31.3. Existing local 1B checkpoint/environment reused;
no installation, model download, conversion, alternate checkout, custom kernel,
privileged collector or GPU trace. Final captures used AC power with its
existing low-power setting enabled (`lowpowermode 1`) and held fixed. Historical M4 data are not pooled.

Run manifests pin checkpoint/config/tokenizer and source hashes. Frozen
pre-execution plan/spec and executable snapshots accompany the accepted r3
captures. Live documentation now includes this outcome; use the snapshots to
recover the registered contract. Source base before this work was repository
commit `31f7b7b`; exact uncommitted runtime source is identified by SHA-256.

## Arm dispositions

| Arm | Change | Numerical records | Whole-model/cache evidence | Decision |
|---|---|---:|---|---|
| C | Unchanged control | Reference | 65 completed single-request/history cases | Matched reference |
| Q | Q/K/V packed | 4,032 pass | First two cases match; third diverges at token 0 | Reject before timing |
| G | Gate/up packed | 2,688 pass | All 65 cases exactly match C, including history/cache metadata | Reject at bounded screen |
| QG | Both | 6,720 pass | First two cases match; third diverges at token 0 | Reject before timing |

All 13,440 numerical records pass finite/allclose checks at the preregistered
atol/rtol=1e-2. These are separate component comparisons across 16 layers,
21 row counts, four input families and each packed projection partition, not
13,440 independent model-quality samples. Inputs use activation dtype fp16,
zeros, ones, seeded bounded random values and captured normalized activations.
The grid includes rows 9/10/11/12/13/17/18/19 and prefill boundaries.

All 480 component/partition byte checks pass. Whole-model logical parameter
bytes remain **695,242,752**; tied output embedding is preserved. Original
projection arrays are removed from the model. Numerical maxima are 0.015625
for Q and 0.03125 for G; nonzero arithmetic differences are permitted by the
numerical diagnostic but do not override the exact-token gate. Actual public
MLX tensor strides are unavailable; contiguity requests/materialization are
recorded without claiming observed strides or measured copy/dispatch counts.

The 65-case gate is 20 natural-EOS fixtures, four forced-128 cases, one
forced-256 case, and 40 history turns: five Experiment 2 scenarios × four turns
× fresh/reuse policies. Candidate fresh is compared with C-fresh and candidate
reuse with C-reuse. Complete rendered histories, exact prefixes, cache offsets,
shapes/types, finite states and terminal EOS/finish behavior are checked. Fresh
and reuse outputs are not required to equal one another.

## QKV divergence diagnostic

Q and QG fail `natural-64-context_question`: C starts with token 40 (`I`),
while the candidate starts with 11439 (`According`). Subsequent lengths/cache
shapes differ as a consequence; that is not independent evidence of a cache
implementation bug. Both arms stop at this first hard failure; later cases
and their timings are intentionally absent.

A separate diagnostic preserves the original cap/prefill/lookahead schedule
and reproduces Q's first-token change. The exposed fp16 log probabilities are:

| Arm | `I` | `According` | Observed selected token |
|---|---:|---:|---|
| C | -1.46875 | -1.46875 | `I` |
| Q | -1.484375 | -1.46875 | `According` |

C has a zero exposed top-two margin; Q has a 0.015625 margin. This supports
sensitivity to finite-precision differences near a tie. Exposed log
probabilities are not unrounded logits; no claim about a unique sampler tie
rule or a particular Metal kernel's causal contribution follows. Exact token
preservation fails regardless of whether either response is semantically useful.

## Clean bounded timing screen: C versus G

One summarization fixture at each length, forced 128 output tokens, one warmup
and five measured repetitions per arm/fixture: **40 measurements, eight warmups**,
plus separately recorded cold requests. Primary timings have no profiler or
telemetry. Each process follows the same ten-second idle rule. The frozen
seed-20261020 order was C/Q/QG/G; filtering rejected Q/QG leaves C/G. Fixture
order, inputs, outputs and measurement source are matched. Both generation and
default streams synchronize at start/end, including queued lookahead; cache
inspection happens after the request timer.

Negative change means lower time or memory. Values below are raw-run medians.

| Target / actual input | C request s | G request s | Request change | C MLX peak MiB | G MLX peak MiB | Peak change |
|---|---:|---:|---:|---:|---:|---:|
| 64 / 64 | 0.769766 | 0.761524 | -1.07% | 797.29 | 811.07 | +1.73% |
| 512 / 510 | 1.040917 | 1.063545 | +2.17% | 1172.13 | 1289.43 | +10.01% |
| 2048 / 2047 | 2.055291 | 2.039667 | -0.76% | 1470.67 | 1640.48 | +11.55% |
| 4096 / 4096 | 3.507463 | 3.503380 | -0.12% | 1558.91 | 1748.79 | +12.18% |

G fails the screen's >=5% request benefit at either short length and exceeds
the 105% request-memory ceiling at targets 512/2048/4096. It stays within the
screen's <=5% request/TTFT median regression limit at every length. Additional
packing peak is 27,787,528 bytes (26.5 MiB), below 512 MiB. Packing leaves
311,427,336 bytes cached in MLX before settling; the identical pre-request
clear-cache policy reduces cached bytes to zero and both arms retain
695,243,528 active bytes before requests. Distinguish allocator cache, active
memory, per-request MLX peak and lifetime process RSS. No swap growth was observed at recorded checks
in either screen process; RSS remains below the 8 GiB capacity ceiling.

The small timing changes are elimination evidence, not precise speed estimates.
Five-sample nearest-rank p95 is the maximum. At 512, TTFT p95 rises 12.15%; the
full-stage tail gate was not exercised, but this diagnostic does not support a
tail improvement. Raw median/p95 TTFT, request and memory comparisons are in the
final screen analysis. No full diverse result or independent repeat is claimed.

The increased peak could be consistent with larger intermediates or copies;
that mechanism was not isolated. Fewer logical projection calls did not
establish fewer GPU dispatches or reduced weight traffic.

## Control quality and harness limitations

Absolute task quality is incomplete and blocks promotion independently of the
performance/memory rejection:

- The 64-target extraction is non-JSON. Very short fixture construction may
  leave insufficient notes; it is not a reliable absolute knowledge test.
- Summaries at 64/512/4096 contain three bullets plus introductory prose. The
  frozen strict three-line validator rejects the introduction; these are not
  demonstrated item-count failures.
- Museum answers correctly say `Mondays`; the strict `Monday` whole-word
  matcher rejects them. These are validator false negatives, not factual errors.
- Extraction validation parses one complete object (or complete fenced block),
  without a semantic key/type schema. Other task semantics remain unmeasured.

C/G exact equivalence proves preservation on these cases, not general quality.
Raw flags are preserved rather than changed after results. A future quality
workload should preregister adequate supplied notes, explicit schemas, fact
alternatives and format semantics. These limitations do not rescue G's failed
benefit/memory gates or Q/QG's exact-token failures.

Two excluded attempts are disclosed. C r1 stopped after system-wide swap grew
90,376,766 bytes (86.19 MiB): 13 outputs exist, only 12 passed capacity checks.
The terminal print also had a relative-path bug after saving its stopped JSON.
The exact r1 worker snapshot is unavailable; its source hash is recorded and
this attempt is excluded. After the user prepared the Mac, C r2 completed, but
Q r2 was blocked before packing by a tuple-versus-JSON-list comparison bug.
Serialized comparison was fixed; a fresh C r3 and all final candidate captures
use the same corrected source. Neither excluded attempt is optimization evidence.

## Review, storage and reproduction

Adversarial reviews cover current primary sources, source preservation,
measurement design, actual correctness data, raw performance arithmetic and
final gate interpretation. The focused CPU suite has **51 passing tests**;
actual GPU gates provide the runtime evidence. A broad initial test glob
accidentally imported the adapter inside the sandbox and temp-file tests were
blocked there; the intended `test*experiment8*.py` suite passes with normal
write access. This was test invocation context, not a model/kernel failure.

All outputs stay in this project. Exact gzip archives preserve selected
intermediate/rejected JSON bytes; the archive receipt records both logical and
container SHA-256s and verified decompression. Do not treat archived paths as
lost evidence. Retained results remain under the 10 MB allowance. No raw GPU
captures or model/runtime files were added. The final validation receipt records
actual output bytes and review/test checks.

```sh
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -B -m unittest discover \
  -s benchmarks -p 'test*experiment8*.py' -q
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -B benchmarks/analyze_experiment8_packed.py \
  --screen benchmarks/results/experiment8-packed-C-screen-20261020-r3.json \
           benchmarks/results/experiment8-packed-G-screen-20261020-r3.json \
  --correctness benchmarks/results/experiment8-packed-C-correctness-20261020-r3.json \
                benchmarks/results/experiment8-packed-G-correctness-20261020-r3.json \
                benchmarks/results/experiment8-packed-Q-correctness-20261020-r3.json.gz \
                benchmarks/results/experiment8-packed-QG-correctness-20261020-r3.json
```

Use unique IDs for any future GPU run. Do not repeat until a preferred outcome
appears. Execution entry points are `run_experiment8_packed.py` and the separate
`diagnose_experiment8_first_token.py`; no site-packages are modified.

Evidence: final C/Q/G/QG correctness r3 files; C/G screen r3; frozen screen
order/execution manifests; `experiment8-packed-screen-analysis-20261020-final.json`;
first-token diagnostic; archive receipt; independent Q/G/final data and method
audits. The dated research/feasibility matrix is
[the execution research note](../../research/experiment-8-packed-projections-execution-research.md).

## Next questions, not new results

The current design exhausts C/Q/G/QG and selects none. The research matrix
records alternatives without claiming exhaustive coverage of inference research.
Useful separately registered follow-ups include decode-only packing versus
prefill-only packing, explicit contiguous splits versus views, fused gate/up +
SwiGLU, or a smaller stateless greedy-sampling seam. New MLX releases or native
runtimes require new controls; M5 accelerator claims do not transfer to M2.
First repair the practical quality contract. No follow-up experiment, larger
model or runtime upgrade was launched by this study.

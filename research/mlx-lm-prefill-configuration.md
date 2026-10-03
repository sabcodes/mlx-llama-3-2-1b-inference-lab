# MLX-LM prefill step: installation and API facts

Accessed 2026-09-25. This note records upstream behavior relevant to the project's Experiment 1; check the installed version before relying on CLI details.

## Verified setup

MLX-LM describes itself as a Python package for Apple silicon and documents `pip install mlx-lm`. For reproducible local work, create an isolated environment in the project (or the user's own project directory), then install there rather than into system Python:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'mlx-lm==0.31.3'
```

The pinned MLX-LM `v0.31.3` package metadata requires Python `>=3.8` and, on macOS, MLX `>=0.31.2`. The upstream README documents pip installation but does not prescribe a virtual-environment workflow; the local venv above is a reproducibility/isolation recommendation. Before running a benchmark, record the actual Python, MLX-LM, MLX, macOS, and architecture versions. The project's Experiment 1 environment recorded MLX-LM `0.31.3`, MLX `0.32.2`, Python `3.14.4`, macOS `15.6.1`, and `arm64`.

## What `prefill_step_size` controls

In MLX-LM `v0.31.3`, `generate_step` accepts `prefill_step_size` with a default of `2048`; it is documented as the step size for processing the prompt. The implementation repeatedly processes up to `min(prefill_step_size, remaining_prompt_tokens)` prompt tokens into the model's KV cache, evaluates the cache state, and clears the MLX cache between chunks. It leaves the final prompt token for the generation step. This is chunked computation of the same token sequence, not prompt truncation: the whole input still contributes to the cache/context. Smaller chunks may lower prefill peak memory at a possible cost in prompt-processing speed; measure this on the target Mac/model/workload.

The supported Python route in `v0.31.3` is `mlx_lm.stream_generate(..., prefill_step_size=N)` (or `mlx_lm.generate(..., prefill_step_size=N)`). `stream_generate` forwards its extra keyword arguments to `generate_step`. The project's benchmark runner already uses that API and exposes its own `--prefill-step-size` flag.

Do not assume the standalone upstream CLI has the same flag in every version. The pinned `v0.31.3` README describes prompt caching and `--max-kv-size` but does not document a `--prefill-step-size` CLI option; the current upstream `main` README documents the flag for `generate`, `chat`, `server`, and `benchmark`, with default 2048. For a version-pinned Experiment 1 reproduction, use the Python API/runner flag and inspect the installed CLI's `--help` before using any upstream CLI option. (The public `main` branch is moving and not a version pin.)

Minimal API sketch for this release:

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
for response in stream_generate(
    model,
    tokenizer,
    prompt,
    max_tokens=128,
    prefill_step_size=2048,
):
    print(response.text, end="", flush=True)
```

Keep the model, prompt tokens/chat template, sampling configuration, output-token cap, cache policy, concurrency, warmup, and repeat count fixed while sweeping only prefill step size. Record actual token count (not approximate characters), TTFT/prefill time and throughput, decode throughput, request latency, and MLX peak memory. Run ordinary timing captures without telemetry; if collecting `powermetrics`, label it as a separate instrumented capture. Compare medians and tail latency per prompt-length/task bucket; choose based on the intended speed/memory tradeoff, then repeat the finalist against the baseline. The project's results are specific to one M2 Pro/model/version/workload and do not establish a universal best value.

## Upstream sources

- MLX-LM README, pinned `v0.31.3` (installation, API, prompt cache): <https://github.com/ml-explore/mlx-lm/blob/v0.31.3/README.md>
- MLX-LM `setup.py`, pinned `v0.31.3` (Python requirement and minimum macOS MLX dependency): <https://github.com/ml-explore/mlx-lm/blob/v0.31.3/setup.py>
- MLX-LM `generate.py`, pinned `v0.31.3` (`generate_step`, `stream_generate`, chunk loop): <https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py>
- MLX-LM README, moving `main` branch (current documented CLI option/default; verify against installed release): <https://github.com/ml-explore/mlx-lm/blob/main/README.md>
- MLX README, pinned `v0.32.2` (MLX installation and framework overview): <https://github.com/ml-explore/mlx/blob/v0.32.2/README.md>

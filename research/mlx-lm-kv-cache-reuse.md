# MLX-LM 0.31.3: exact KV-prefix reuse across chat turns

**Scope:** source-level review of the tagged `v0.31.3` implementation only. This is not a benchmark or a claim about other MLX-LM versions.

## Bottom line

MLX-LM 0.31.3 can reuse KV state for a prompt prefix, but the low-level Python generation calls do **not** find matching prefixes for you. `generate_step` and `stream_generate` accept a caller-owned `prompt_cache` and update it in place. Exact token-prefix matching and returning only the uncached suffix are implemented by `LRUPromptCache.fetch_nearest_cache`; the bundled `mlx_lm.server` uses that cache automatically.

For Experiment 2, either exercise `mlx_lm.server`'s built-in cache behavior, or explicitly reproduce its sequence: tokenize the complete conversation with the same model chat template, ask an `LRUPromptCache` for the longest reusable prefix, then pass the returned suffix and cache to `stream_generate`. Passing a non-empty cache alongside the full prompt is not a cache hit: `generate_step` processes the provided prompt against that cache and will append those tokens again.

## What the tagged code supports

| Capability | Evidence in v0.31.3 | What it means |
|---|---|---|
| Caller-provided generation cache | [`generate_step`](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/generate.py#L295-L378) accepts `prompt_cache`; its docstring says it is precomputed and updated in place. [`stream_generate`](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/generate.py#L621-L669) forwards remaining keyword arguments to `generate_step`. | The caller owns cache lifecycle and must only send tokens not already represented in cache. |
| Model-appropriate cache creation | [`make_prompt_cache`](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/models/cache.py#L14-L38) delegates to `model.make_cache()` when present, otherwise creates per-layer `KVCache`s (or rotating caches if max size is supplied). | Prefer this helper over hand-constructing cache objects. |
| Exact longest-prefix lookup | [`LRUPromptCache.fetch_nearest_cache`](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/models/cache.py#L1548-L1566), backed by [`PromptTrie.search`](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/models/cache.py#L1460-L1499). | Matching is on token IDs and a model key, not on text similarity. It returns `(cache, remaining_tokens)`; no match returns `(None, all_tokens)`. It deep-copies fetched cache state so a generation can extend it independently. |
| Prefix trimming | [`can_trim_prompt_cache` / `trim_prompt_cache`](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/models/cache.py#L81-L103), plus cache-specific `is_trimmable`/`trim` implementations (e.g. [`KVCache`](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/models/cache.py#L302-L333)). | Trimming is in-place and only proceeds if every layer cache says it is trimmable. Fixed/sliding-window cache behavior can constrain reuse once its capacity is reached; do not assume every model cache can rewind. |
| Built-in server prefix reuse | [`server._serve_single`](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/server.py#L854-L907) fetches nearest cache and passes only `rest` to `stream_generate`; it then appends generated token IDs to the cache key and inserts the resulting cache ([lines 926–940](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/server.py#L926-L940)). | The server is the clearest package-provided end-to-end implementation to benchmark if the project can use its request interface. |
| Persisting a cache | [`save_prompt_cache` / `load_prompt_cache`](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/models/cache.py#L40-L79). | Serialization is supported, but persistence does not itself identify a matching prompt. The caller must track compatible model/tokenizer/config and token-prefix metadata. |

## API boundary: supported versus caller logic

**Direct `stream_generate` usage:** supported for receiving a cache through `prompt_cache=`. It does not look up the cache against the supplied prompt, compare tokens, or remove a cached prefix. The generator processes the prompt argument and mutates the passed cache. Treat `prompt` as the exact uncached suffix when reusing a cache.

**Prefix lookup:** `LRUPromptCache` provides a source-implemented exact token-prefix matcher, and the server uses it. The repo code exposes the class from `mlx_lm.models.cache`; however, the release source does not present `LRUPromptCache` as a separately versioned public API contract. Pin MLX-LM and test this behavior when upgrading.

**Chat API behavior:** the server converts the supplied `messages` to one token sequence using the configured `apply_chat_template` (`server.py` [lines 494–518](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/server.py#L494-L518)), then does prefix lookup on those IDs. That is materially different from the low-level Python API, where the application must tokenize/template and select the suffix/cache itself.

The shipped example [`mlx_lm/examples/chat.py`](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.31.3/mlx_lm/examples/chat.py#L2-L50) demonstrates reusing a `make_prompt_cache(model)` object across calls to `generate(..., prompt_cache=prompt_cache)` and saving/loading it. It does **not** demonstrate a full-history prompt comparison or an automatic longest-prefix lookup; don't infer those behaviors from the example alone. In particular, do not pass a full conversation token sequence into a cache that already contains its prefix.

## Chat-template and append-only caveats for a valid experiment

1. Build each request from the same model tokenizer and chat-template configuration, using the full message history and the normal assistant-generation boundary (the server uses `add_generation_prompt=True`). Keep tool definitions, system content, template kwargs, tokenizer/model revision, and any adapter/model key fixed.
2. Compare token IDs, not rendered strings. For turn *n+1*, calculate the longest common token prefix between its fully templated prompt and the cache key/state after turn *n*. Different whitespace, special-token insertion, role separators, template flags, tool schemas, or tokenization boundaries can shorten the hit even when the text looks almost identical.
3. Responses are generated token IDs, while a later chat request often reconstructs the assistant message from decoded text and re-applies the template. That re-encoded prefix is not guaranteed to be identical to generated IDs plus end-of-turn markers. Preserve actual generated token IDs and verify how they align with the next fully templated request; measure the actual cached-token count rather than assuming the entire previous turn hits.
4. Prefix reuse is naturally append-only only while the earlier conversation is unchanged. Editing/deleting an old message, changing the system prompt/template/tools, changing model or adapter, or truncating history can invalidate the old suffix. An exact prefix cache can still reuse the unchanged prefix, but only if the token-level common prefix is checked and the remaining tail is recomputed.
5. Cache rewind is conditional. `LRUPromptCache` limits reuse from a longer cached sequence to caches that report trimmable; `KVCache.trim()` decreases its logical offset, while other model cache types may refuse trimming or behave differently near a bounded context window. Record cache type, logical size, matched-token count, suffix-token count, and fallback reason where possible.
6. Ensure the prompt suffix contains at least one token. The cache lookup intentionally caps the reusable common prefix to at most `len(tokens)-1` for a longer-match/trim path, preserving a final token for model processing; inspect the returned suffix rather than assuming every token was consumed from cache.
7. Fully consume the generation iterator before treating a turn's cache as complete. Stop conditions, cancellation, and early iterator exit may leave an incomplete turn/cache state; the server has its own handling and should not be assumed equivalent to arbitrary caller behavior.

## Implications for Experiment 2

- A fair control should use the same exact tokenized multi-turn requests and generation settings but create a fresh cache per request/turn, so the full prompt is recomputed. Treatment uses the longest exact token-prefix cache and only evaluates the returned suffix.
- Keep `prefill_step_size`, model, tokenizer/template, seed/sampling, max output, conversation messages, and turn ordering fixed across arms. For repeated benchmark runs, clear or recreate the treatment cache at the same declared scope (e.g. start of each conversation repetition) so warm state does not leak between repetitions unless persistent cross-conversation reuse is explicitly the subject.
- Verify correctness by comparing the emitted output tokens (prefer deterministic greedy decoding for the test) between the fresh-cache and reused-prefix arms. Differences can indicate suffix-boundary, cache-offset, template, or generation-state mistakes, not just performance noise.
- Capture requested prompt tokens, cached tokens, newly-prefilled tokens, TTFT, prefill throughput, decode throughput, total turn latency, and cache bytes/size. “Cached prompt tokens” must be measured from the actual prefix lookup result; do not infer it from prior turn length.
- If using the server's automatic cache path, test the same-session/ordered requests that populate then reuse the cache, and read its reported `usage.prompt_tokens_details.cached_tokens` where non-streaming responses expose it. If implementing direct Python lookup, log the `LRUPromptCache` return values and use precisely the returned `rest` argument with `stream_generate`.

## Source links

All links above are pinned to the upstream `v0.31.3` tag:

- [`generate.py`](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py)
- [`models/cache.py`](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py)
- [`server.py`](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/server.py)
- [`examples/chat.py`](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/examples/chat.py)

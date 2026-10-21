"""Experiment-local, lossless row packing for the pinned MLX-LM Llama model.

``pack_model(model, arm, verify=None)`` mutates one layer family at a time.
The optional callback is ``verify(layer_index, family, originals, packed)``;
family is ``Q`` or ``G``, originals is a tuple in output-row order, and packed
is a ``PackedLinear``. ``packed(x)`` returns the concatenated output and
``packed.split(x)`` returns the original output partitions. The callback runs
before original modules are replaced and may return JSON-serializable scalar
verification evidence. It must not retain original modules or tensor outputs.

This adapter never constructs QuantizedLinear, re-quantizes, saves weights, or
changes the tied embedding/output head. Runtime numerical tests live in the
runner, not in the timed request path.
"""

import ast
import hashlib
import inspect
import json
import textwrap

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import llama
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention


_SOURCE_CONTRACT = {
    "Attention": "e9b1bb074e500a562f8c4a147c5d1f23f26e2505c6faa84868991214d4daee2b",
    "MLP": "ff48284dcfcde60a27d4143defacec431bff19ac5251afbb579224907fcdfce8",
}


def _require(condition, message):
    if not condition:
        raise ValueError(f"Experiment 8 packing rejected: {message}")


def _source_contract():
    """Fail closed when the installed downstream implementation changes."""
    for name, expected in _SOURCE_CONTRACT.items():
        source = textwrap.dedent(inspect.getsource(getattr(llama, name).__call__))
        node = ast.parse(source).body[0]
        digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        _require(digest == expected, f"installed {name}.__call__ source changed")


def _array_description(value, contiguity="not exposed by pinned public MLX API"):
    # Pinned MLX has no public strides/flags API. Do not invent observations.
    strides = getattr(value, "strides", None)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nbytes": value.nbytes,
        "strides": list(strides) if strides is not None else None,
        "contiguity": contiguity,
        "layout_observation": "strides unavailable in pinned public MLX API" if strides is None else "strides exposed",
    }


def _parameter_arrays(model):
    return [value for _, value in tree_flatten(model.parameters())]


class PackedLinear(nn.Module):
    """Existing affine packed rows joined once; no random initialization."""

    def __init__(self, originals):
        super().__init__()
        originals = tuple(originals)
        _require(len(originals) in (2, 3), "expected two or three projections")
        first = originals[0]
        _require(all(type(p) is nn.QuantizedLinear for p in originals), "unsupported linear type")
        _require(all("bias" not in p for p in originals), "ordinary linear bias is unsupported")
        self.group_size, self.bits, self.mode = first.group_size, first.bits, first.mode
        _require((self.group_size, self.bits, self.mode) == (64, 4, "affine"), "expected affine-4/group-64")
        self.input_dims = first.weight.shape[1] * 32 // self.bits
        self.output_dims = tuple(p.weight.shape[0] for p in originals)
        self.split_indices = tuple(sum(self.output_dims[:i]) for i in range(1, len(originals)))
        components = ("weight", "scales", "biases")
        for projection in originals:
            _require((projection.group_size, projection.bits, projection.mode) ==
                     (self.group_size, self.bits, self.mode), "quantization fields differ")
            _require(projection.weight.ndim == 2, "weight must be a matrix")
            _require(projection.weight.shape[1] * 32 // self.bits == self.input_dims, "input dimensions differ")
            _require(projection.weight.dtype == mx.uint32, "expected uint32 packed weights")
            rows = projection.weight.shape[0]
            for name in components:
                value = projection.get(name)
                _require(isinstance(value, mx.array), f"missing affine component {name}")
                _require(value.ndim == 2 and value.shape[0] == rows, f"invalid {name} rows")
                _require(value.dtype == getattr(first, name).dtype, f"{name} dtypes differ")
                _require(value.shape[1:] == getattr(first, name).shape[1:], f"{name} column shapes differ")
            _require(projection.scales.shape == (rows, self.input_dims // self.group_size), "invalid scale dimensions")
            _require(projection.biases.shape == projection.scales.shape, "invalid affine bias dimensions")
            _require(projection.biases.dtype == projection.scales.dtype, "affine bias/scale dtype differs")

        for name in components:
            setattr(self, name, mx.contiguous(mx.concatenate([getattr(p, name) for p in originals], axis=0)))
        mx.eval(self.weight, self.scales, self.biases)
        self.freeze()

        checks = []
        offset = 0
        for index, projection in enumerate(originals):
            end = offset + self.output_dims[index]
            entries = {}
            for name in components:
                source = getattr(projection, name)
                piece = getattr(self, name)[offset:end]
                equal = bool(mx.array_equal(source.view(mx.uint8), piece.view(mx.uint8)).item())
                _require(equal, f"{name} byte mismatch at partition {index}")
                entries[name] = {
                    "bitwise_equal": equal,
                    "source": _array_description(source),
                    "packed_slice": _array_description(piece, "row slice of materialized mx.contiguous result"),
                }
            checks.append({"partition": index, "start": offset, "end": end, "components": entries})
            offset = end
        before = sum(getattr(p, name).nbytes for p in originals for name in components)
        after = sum(getattr(self, name).nbytes for name in components)
        _require(before == after, "packed parameter byte count differs")
        # Only JSON metadata survives; there is no reference to source modules.
        self.check = {
            "group_size": self.group_size, "bits": self.bits, "mode": self.mode,
            "transpose": True, "input_dims": self.input_dims,
            "output_dims": list(self.output_dims), "split_indices": list(self.split_indices),
            "source_parameter_bytes": before, "packed_parameter_bytes": after,
            "components": {name: _array_description(getattr(self, name), "mx.contiguous(..., allow_col_major=False) requested and evaluated") for name in components},
            "partitions": checks,
        }

    def __call__(self, x):
        return mx.quantized_matmul(
            x, self.weight, scales=self.scales, biases=self.biases,
            transpose=True, group_size=self.group_size, bits=self.bits, mode=self.mode,
        )

    def split(self, x):
        return tuple(mx.split(self(x), self.split_indices, axis=-1))


class PackedAttention(nn.Module):
    def __init__(self, original, packed):
        super().__init__()
        self.n_heads = original.n_heads
        self.n_kv_heads = original.n_kv_heads
        self.head_dim = original.head_dim
        self.scale = original.scale
        self.rope = original.rope
        self.o_proj = original.o_proj
        self.qkv_proj = packed
        self._training = original.training
        self.freeze()

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape

        queries, keys, values = self.qkv_proj.split(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class PackedMLP(nn.Module):
    def __init__(self, original, packed):
        super().__init__()
        self.down_proj = original.down_proj
        self.gate_up_proj = packed
        self._training = original.training
        self.freeze()

    def __call__(self, x):
        gate, up = self.gate_up_proj.split(x)
        return self.down_proj(swiglu(gate, up))


def pack_model(model, arm, verify=None):
    """Pack C/Q/G/QG in place; return byte/layout and callback evidence.

    Stop at the first invalid family or callback exception. The model may be
    partially adapted on failure and must be discarded by the calling runner.
    """
    _require(arm in ("C", "Q", "G", "QG"), f"unknown arm {arm}")
    _source_contract()
    _require(type(model) is llama.Model, "expected pinned Llama Model")
    _require(model.args.tie_word_embeddings and "lm_head" not in model, "expected tied output head")
    _require(len(model.layers) == 16 and model.args.hidden_size == 2048, "unexpected model dimensions")
    embedding = model.model.embed_tokens
    before_arrays = _parameter_arrays(model)
    before_bytes = sum(a.nbytes for a in before_arrays)
    del before_arrays
    checks = []
    for index, layer in enumerate(model.layers):
        families = []
        if "Q" in arm:
            families.append(("Q", "self_attn", ("q_proj", "k_proj", "v_proj"), (2048, 512, 512), PackedAttention))
        if "G" in arm:
            families.append(("G", "mlp", ("gate_proj", "up_proj"), (8192, 8192), PackedMLP))
        for family, attribute, names, expected_dims, wrapper in families:
            original = getattr(layer, attribute)
            _require(type(original) is (llama.Attention if family == "Q" else llama.MLP), "unexpected source module")
            originals = tuple(getattr(original, name) for name in names)
            packed = PackedLinear(originals)
            _require(packed.input_dims == 2048 and packed.output_dims == expected_dims, "unexpected family projection dimensions")
            check = {"layer": index, "family": family, **packed.check}
            if verify is not None:
                evidence = verify(index, family, originals, packed)
                # Reject tensors/modules in evidence, including nested ones.
                json.dumps(evidence, allow_nan=False)
                check["verification"] = evidence
            removed_ids = {id(getattr(p, name)) for p in originals for name in ("weight", "scales", "biases")}
            setattr(layer, attribute, wrapper(original, packed))
            _require(not any(id(a) in removed_ids for a in _parameter_arrays(model)),
                     "original projection arrays remain in model")
            checks.append(check)
            del original, originals, packed
    after_arrays = _parameter_arrays(model)
    after_bytes = sum(a.nbytes for a in after_arrays)
    _require(before_bytes == after_bytes, "whole-model parameter byte count changed")
    _require(model.model.embed_tokens is embedding and "lm_head" not in model, "tied output head changed")
    for check in checks:
        check["whole_model_parameter_bytes_before"] = before_bytes
        check["whole_model_parameter_bytes_after"] = after_bytes
        check["original_projection_arrays_retained_in_model"] = False
        check["tied_head_preserved"] = True
    return checks

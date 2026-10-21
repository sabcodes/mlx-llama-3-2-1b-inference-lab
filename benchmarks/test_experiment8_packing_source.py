"""CPU-only checks that packed wrappers preserve pinned downstream operations.

Run before inference; deliberately do not import MLX or allocate GPU work.
"""
import ast
import hashlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "benchmarks/experiment8_packing.py"
LLAMA = ROOT / ".venv/lib/python3.14/site-packages/mlx_lm/models/llama.py"


def method(path, class_name, method_name="__call__"):
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def dump(node):
    return ast.dump(node, include_attributes=False)


class DownstreamPreservation(unittest.TestCase):
    def test_attention_changes_only_projection_assignment(self):
        control = method(LLAMA, "Attention")
        packed = method(ADAPTER, "PackedAttention")
        self.assertEqual(dump(control.body[0]), dump(packed.body[0]))
        self.assertEqual([dump(n) for n in control.body[2:]], [dump(n) for n in packed.body[2:]])
        self.assertEqual(dump(packed.body[1].value), dump(ast.parse("self.qkv_proj.split(x)", mode="eval").body))

    def test_mlp_preserves_swiglu_and_down_projection(self):
        control = method(LLAMA, "MLP")
        packed = method(ADAPTER, "PackedMLP")
        expected = ast.parse("return self.down_proj(swiglu(gate, up))").body[0]
        self.assertEqual(dump(packed.body[-1]), dump(expected))
        original_expected = ast.parse("return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))").body[0]
        self.assertEqual(dump(control.body[0]), dump(original_expected))
        self.assertEqual(dump(packed.body[0]), dump(ast.parse("gate, up = self.gate_up_proj.split(x)").body[0]))

    def test_runtime_contract_matches_installed_methods(self):
        tree = ast.parse(ADAPTER.read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_SOURCE_CONTRACT" for t in n.targets))
        hashes = ast.literal_eval(assignment.value)
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256(dump(method(LLAMA, name)).encode()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()

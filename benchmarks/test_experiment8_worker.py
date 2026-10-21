"""CPU-only contract checks for Experiment 8; never import MLX or run inference."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment8_packed as worker
import diagnose_experiment8_first_token as diagnostic


def generation_record():
    return {
        "output_token_ids": [11, 12, 99],
        "generated_tokens": 3,
        "response_token_count": 3,
        "eos_position_zero_based": 2,
        "finish_reason": "stop",
        "cache": [{"type": "KVCache", "offset": 10, "shapes": [[1, 8, 10, 64]], "bytes": 20480}],
        "generated_text_sha256": "same-output",
    }


def state(**changes):
    result = {
        "swap_used_bytes": 100,
        "process_rss_bytes": 2**30,
        "process_peak_rss_bytes": 2**30,
        "power_config": "lowpowermode 0",
        "supply": "Now drawing from AC Power",
    }
    result.update(changes)
    return result


class ArchivedReferenceTests(unittest.TestCase):
    def read_virtual(self, requested, files):
        with patch.object(Path, "is_file", lambda path: path in files), \
                patch.object(Path, "read_bytes", lambda path: files[path]):
            return diagnostic.read_reference(requested)

    def test_plain_and_explicit_gzip_preserve_original_logical_bytes(self):
        raw = b'{  "arm": "Q", "status": "rejected" }\n'
        plain = Path("/virtual/control.json")
        archived = Path("/virtual/control.json.gz")
        files = {plain: raw, archived: gzip.compress(raw, mtime=0)}
        for path in (plain, archived):
            with self.subTest(path=path):
                actual_path, logical, payload = self.read_virtual(path, files)
                self.assertEqual(actual_path, path)
                self.assertEqual(logical, raw)
                self.assertEqual(hashlib.sha256(logical).hexdigest(),
                                 hashlib.sha256(raw).hexdigest())
                self.assertEqual(payload, {"arm": "Q", "status": "rejected"})

    def test_missing_plain_reference_falls_back_to_gzip(self):
        plain = Path("/virtual/candidate.json")
        archived = Path("/virtual/candidate.json.gz")
        raw = b'{"arm":"Q"}'
        actual_path, logical, payload = self.read_virtual(
            plain, {archived: gzip.compress(raw, mtime=0)})
        self.assertEqual(actual_path, archived)
        self.assertEqual(logical, raw)
        self.assertEqual(payload["arm"], "Q")

    def test_existing_plain_reference_takes_precedence_over_archive(self):
        plain = Path("/virtual/candidate.json")
        archived = Path("/virtual/candidate.json.gz")
        files = {plain: b'{"arm":"C"}',
                 archived: gzip.compress(b'{"arm":"Q"}', mtime=0)}
        self.assertEqual(self.read_virtual(plain, files)[2]["arm"], "C")

    def test_reference_payload_must_be_a_json_object(self):
        path = Path("/virtual/bad.json.gz")
        with self.assertRaisesRegex(ValueError, "must be an object"):
            self.read_virtual(path, {path: gzip.compress(b"[]", mtime=0)})


class SerializedContractTests(unittest.TestCase):
    def test_json_roundtrip_normalizes_nested_tuples_without_changing_contract(self):
        live = {"numerical_shapes": worker.SHAPES,
                "nested": {"quantization": (64, 4, "affine")},
                "greedy": True}
        persisted = json.loads(json.dumps(live))
        self.assertNotEqual(live, persisted)
        self.assertEqual(worker.digest(live), worker.digest(persisted))

    def test_serialized_contract_still_rejects_changed_values_and_array_order(self):
        control = {"numerical_shapes": (1, 2, 31, 32), "greedy": True}
        for candidate in (
            {"numerical_shapes": [1, 2, 31, 33], "greedy": True},
            {"numerical_shapes": [2, 1, 31, 32], "greedy": True},
            {"numerical_shapes": [1, 2, 31, 32], "greedy": False},
        ):
            with self.subTest(candidate=candidate):
                self.assertNotEqual(worker.digest(control), worker.digest(candidate))


class CompareTests(unittest.TestCase):
    def test_identical_output_and_cache_metadata_pass(self):
        control = generation_record()
        self.assertTrue(worker.compare(copy.deepcopy(control), control)["passed"])

    def test_first_token_difference_has_control_candidate_orientation(self):
        control = generation_record()
        candidate = copy.deepcopy(control)
        candidate["output_token_ids"][1] = 77
        comparison = worker.compare(candidate, control)
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["first_token_difference"],
                         {"index": 1, "control_token": 12, "candidate_token": 77})

    def test_token_length_difference_reports_missing_candidate(self):
        control = generation_record()
        candidate = copy.deepcopy(control)
        candidate["output_token_ids"].pop()
        comparison = worker.compare(candidate, control)
        self.assertEqual(comparison["first_token_difference"],
                         {"index": 2, "control_token": 99, "candidate_token": None})

    def test_terminal_and_cache_contract_fields_cannot_silently_differ(self):
        for key, changed in (("generated_tokens", 2), ("response_token_count", 2),
                             ("eos_position_zero_based", None), ("finish_reason", "length"),
                             ("generated_text_sha256", "other")):
            with self.subTest(key=key):
                control = generation_record()
                candidate = copy.deepcopy(control)
                candidate[key] = changed
                self.assertIn(key, worker.compare(candidate, control)["different_fields"])
        control = generation_record()
        candidate = copy.deepcopy(control)
        candidate["cache"][0]["offset"] += 1
        self.assertIn("cache", worker.compare(candidate, control)["different_fields"])


class QualityTests(unittest.TestCase):
    def test_complete_json_object_and_complete_fenced_block_pass(self):
        for text in ('{"hours":"nine to six"}', '```json\n{"hours":"nine to six"}\n```'):
            with self.subTest(text=text):
                self.assertTrue(worker.quality(text, "extraction")["complete_json_object"])

    def test_json_substrings_arrays_truncations_and_multiple_blocks_fail(self):
        for text in ('Here is {"hours":"nine to six"}', '[{"hours":6}]', '{"hours":',
                     '```json\n{"a":1}\n```\nextra', '{"a":1} {"b":2}', '{}'):
            with self.subTest(text=text):
                self.assertFalse(worker.quality(text, "extraction")["complete_json_object"])

    def test_exact_three_bullets_pass_but_extra_prose_and_missing_items_fail(self):
        self.assertTrue(worker.quality("- one\n* two\n• three\n", "summarization")["exactly_three_bullets"])
        for text in ("- one\n- two", "- one\n- two\n- three\n- four",
                     "Summary:\n- one\n- two\n- three", "one\ntwo\nthree"):
            with self.subTest(text=text):
                self.assertFalse(worker.quality(text, "summarization")["exactly_three_bullets"])

    def test_fact_tokens_match_normalized_numbers_and_whole_words(self):
        self.assertTrue(worker.quality("Book at least 2 days ahead.", "history", "two days")["required_fact"])
        self.assertTrue(worker.quality("Only when no EVENT is scheduled.", "history", "event")["required_fact"])
        for text, expected in (("An eventual opening.", "event"),
                               ("Book 12 days ahead.", "two days"),
                               ("Located at the north masthead.", "north mast")):
            with self.subTest(text=text, expected=expected):
                self.assertFalse(worker.quality(text, "history", expected)["required_fact"])


class CapacityTests(unittest.TestCase):
    def check(self, current, initial=None):
        with patch.object(worker, "machine_state", return_value=current):
            return worker.capacity_check(initial if initial is not None else state())

    def test_unchanged_state_and_exact_rss_limit_pass(self):
        self.assertEqual(self.check(state()), state())
        self.check(state(process_peak_rss_bytes=8 * 2**30))

    def test_swap_growth_and_peak_rss_limit_stop(self):
        for current, message in ((state(swap_used_bytes=101), "system swap grew"),
                                 (state(process_peak_rss_bytes=8 * 2**30 + 1), "8 GiB")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.check(current)

    def test_missing_start_or_end_swap_is_unmeasured(self):
        for current, initial in ((state(swap_used_bytes=None), state()),
                                 (state(), state(swap_used_bytes=None))):
            with self.subTest(current=current, initial=initial):
                with self.assertRaisesRegex(RuntimeError, "capacity_unmeasured"):
                    self.check(current, initial)

    def test_missing_current_rss_is_unmeasured(self):
        with self.assertRaisesRegex(RuntimeError, "capacity_unmeasured"):
            self.check(state(process_rss_bytes=None))

    def test_changed_power_or_supply_invalidates_capture(self):
        for changes in ({"power_config": "lowpowermode 1"}, {"supply": "Battery Power"}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(RuntimeError, "power_state_changed"):
                    self.check(state(**changes))


class FakeCache:
    def __init__(self, offset):
        self.offset = offset

    @property
    def state(self):
        return [types.SimpleNamespace(shape=(1, 8, self.offset, 64))] * 2

    @property
    def nbytes(self):
        return self.offset * 2048


class GenerationOffsetTests(unittest.TestCase):
    def invoke(self, initial=0, actual_adjustment=0, terminal="stop", forced=False):
        tokens, output = [1, 2, 3], [11, 12, 99 if terminal == "stop" else 13]
        caches = [FakeCache(initial) for _ in range(16)]
        package = types.ModuleType("mlx_lm")
        cache_module = types.ModuleType("mlx_lm.models.cache")
        sample_module = types.ModuleType("mlx_lm.sample_utils")
        cache_module.make_prompt_cache = lambda model: caches
        sample_module.make_sampler = lambda **kwargs: object()
        sample_module.make_logits_processors = lambda **kwargs: []

        def stream_generate(model, tokenizer, prompt, **kwargs):
            for cache in kwargs["prompt_cache"]:
                cache.offset = initial + len(tokens) + len(output) + actual_adjustment
            for index, token in enumerate(output):
                yield types.SimpleNamespace(token=token, text="" if token == 99 else str(token),
                                            finish_reason=terminal if index == len(output)-1 else None,
                                            generation_tokens=index+1, prompt_tps=1.0, generation_tps=1.0)
        package.stream_generate = stream_generate
        modules = {"mlx_lm": package, "mlx_lm.models.cache": cache_module,
                   "mlx_lm.sample_utils": sample_module}
        mx = types.SimpleNamespace(reset_peak_memory=lambda: None)
        memory = {"peak_bytes": 100, "active_bytes": 80, "cached_bytes": 0,
                  "process_peak_rss_bytes": 1000}
        with patch.dict(sys.modules, modules), patch.object(worker, "sync"), \
                patch.object(worker, "memory", return_value=memory):
            return worker.generation(object(), types.SimpleNamespace(eos_token_ids={99}),
                                     mx, tokens, cap=3, forced=forced, cache=caches)[0]

    def test_fresh_terminal_eos_includes_eos_lookahead_in_expected_offset(self):
        result = self.invoke()
        self.assertEqual(result["expected_cache_offset"], 6)
        self.assertEqual(result["eos_position_zero_based"], 2)
        self.assertEqual(result["visible_non_eos_count"], 2)

    def test_reused_prefix_offset_is_added_to_suffix_and_emitted_tokens(self):
        self.assertEqual(self.invoke(initial=7)["expected_cache_offset"], 13)

    def test_consistent_but_off_by_one_cache_offsets_are_rejected(self):
        for adjustment in (-1, 1):
            with self.subTest(adjustment=adjustment):
                with self.assertRaisesRegex(AssertionError, "cache_offset_mismatch"):
                    self.invoke(actual_adjustment=adjustment)

    def test_forced_contract_requires_length_finish_and_no_eos(self):
        self.assertEqual(self.invoke(terminal="length", forced=True)["finish_reason"], "length")
        with self.assertRaisesRegex(AssertionError, "forced_output_contract_failed"):
            self.invoke(terminal="stop", forced=True)


if __name__ == "__main__":
    unittest.main()

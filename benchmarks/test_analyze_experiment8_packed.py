"""Synthetic CPU-only checks for Experiment 8 evidence and decision gates."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_experiment8_packed as a


def generation(prompt=64, forced=False, cap=128, request=1., ttft=.1, peak=1.):
    tokens = [2] * cap if forced else [2, 3, 128009]
    offset = prompt + len(tokens)
    return {"output_token_ids": tokens, "generated_tokens": len(tokens),
        "response_token_count": len(tokens), "visible_non_eos_count": len(tokens) - (not forced),
        "eos_position_zero_based": None if forced else len(tokens)-1,
        "finish_reason": "length" if forced else "stop", "generated_text": "synthetic",
        "generated_text_sha256": hashlib.sha256(b"synthetic").hexdigest(),
        "cached_token_history_sha256": a.digest([prompt, *tokens]),
        "time_to_first_token_seconds": ttft, "request_seconds": request,
        "mlx_peak_memory_gb": peak, "expected_cache_offset": offset, "cache_finite": True,
        "memory": {"peak_bytes": peak * 1e9, "process_peak_rss_bytes": 2**30},
        "cache": [{"type": "KVCache", "offset": offset,
                   "shapes": [[1,8,offset,64], [1,8,offset,64]], "bytes": 1000} for _ in range(16)]}


def packing(arm):
    entries = []
    for layer in range(16):
        for family in ("Q", "G"):
            if family not in arm:
                continue
            dims = [2048,512,512] if family == "Q" else [8192,8192]
            entries.append({"layer": layer, "family": family, "output_dims": dims, "input_dims":2048,
                "group_size":64, "bits":4, "mode":"affine", "transpose": True,
                "source_parameter_bytes":sum(dims)*1152, "packed_parameter_bytes":sum(dims)*1152,
                "components":{name:{"shape":[sum(dims),256 if name=="weight" else 32],
                    "dtype":"uint32" if name=="weight" else "bfloat16",
                    "nbytes":sum(dims)*(1024 if name=="weight" else 64)} for name in ("weight","scales","biases")},
                "whole_model_parameter_bytes_before":100_000, "whole_model_parameter_bytes_after":100_000,
                "original_projection_arrays_retained_in_model":False, "tied_head_preserved":True,
                "split_indices":[sum(dims[:i]) for i in range(1,len(dims))],
                "partitions":[{"partition":i, "start":sum(dims[:i]), "end":sum(dims[:i+1]),
                    "components":{name:{"bitwise_equal":True, "source":{"shape":[n,256 if name=="weight" else 32],"dtype":"uint32" if name=="weight" else "bfloat16","nbytes":n*(1024 if name=="weight" else 64)},
                        "packed_slice":{"shape":[n,256 if name=="weight" else 32],"dtype":"uint32" if name=="weight" else "bfloat16","nbytes":n*(1024 if name=="weight" else 64)}} for name in ("weight","scales","biases")}}
                    for i,n in enumerate(dims)],
                "verification":{"passed": True, "checks":[{"layer":layer,"family":family,"rows":rows,"input":kind,
                    "component":j,"finite":True,"allclose":True,"max_abs":0.,"rms":0.,"output_shape":[1,rows,n]}
                    for rows in a.ROWS for kind in ("zeros","ones","random","activation") for j,n in enumerate(dims)]}})
    return entries


def payload(arm="C", stage="screen", multiplier=1., slot=0):
    start = datetime(2026,10,20,tzinfo=timezone.utc) + timedelta(hours=slot)
    manifest = {key:"a"*64 for key in ("mlx_lm.models.llama", "mlx_lm.generate", "mlx.nn.layers.quantized",
        "experiments/experiment-8-packed-projections-plan.md", "experiments/experiment-8-packed-projections-spec.md",
        "benchmarks/run_experiment8_packed.py", "benchmarks/experiment8_packing.py", "benchmarks/run_baseline.py",
        "benchmarks/fixtures/experiment2-sessions.json", "models/existing/model.safetensors", "models/existing/config.json",
        "models/existing/tokenizer.json", "models/existing/tokenizer_config.json")}
    state = {"power_config":"normal", "supply":"AC Power", "swap_used_bytes":100,
             "process_rss_bytes":2**30,"process_peak_rss_bytes":2**30}
    p = {"experiment":"experiment8-packed-projections", "arm":arm,"stage":stage,
        "workload_id":"exp8-packed-" + {"screen":"screen-v1","full":"diverse-v1","correctness":"correctness-v1"}[stage],
        "started_at_utc":start.isoformat(), "completed_at_utc":(start+timedelta(minutes=10)).isoformat(),
        "status":"pass", "environment":{"mlx":"0.32.2","mlx_lm":"0.31.3","device":{"device_name":"Apple M2 Pro"}},
        "settings":{"date":"04 Oct 2026","prefill_step_size":2048,"greedy":True,"fresh_primary_cache":True,
            "concurrency":1,"numerical_shapes":list(a.ROWS),"numerical_seed":20261020},
        "manifest":manifest, "parameter_bytes_before":100_000,"parameter_bytes_after":100_000,
        "state_start":copy.deepcopy(state),"state_last":copy.deepcopy(state),"state_end":copy.deepcopy(state),
        "memory_loaded":{"peak_bytes":10**9},"memory_packing":{"peak_bytes":10**9},
        "packing":packing(arm) if arm != "C" else [],"absolute_quality_pass":True,
        "absolute_quality_scope":"Complete synthetic task assertions evaluated.","checks":{},"cases":[]}
    if stage == "correctness":
        for key in sorted(a.expected_correctness_ids()):
            p["checks"][key] = {"prompt_sha256":"b"*64,"generation":generation(forced=key.startswith("forced-"),
                cap=256 if key == "forced-2048-256" else 128),"quality":{"synthetic_fact":True}}
    else:
        for key in sorted(a.expected_case_ids(stage)):
            n = int(key.split("-")[1])
            gen = generation(n,stage=="screen",request=multiplier)
            p["cases"].append({"id":key,"target_prompt_tokens":n,"actual_prompt_tokens":n,"prompt_sha256":"b"*64,
                "warmup":copy.deepcopy(gen),"runs":[copy.deepcopy(gen) for _ in range(5 if stage=="screen" else 10)],
                "summary":{"deliberately":"ignored"}})
    return p


class ArchivedPayloads(unittest.TestCase):
    def test_plain_and_gzip_readers_preserve_identical_evidence(self):
        evidence = {"arm": "Q", "status": "rejected", "failure": {"message": "numeric Δ"},
                    "raw": [0, 1.23456789012345, {"tokens": [128009], "finite": False}]}
        raw = (json.dumps(evidence, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        with tempfile.TemporaryDirectory(prefix="experiment8-reader-test-") as folder:
            plain = Path(folder) / "result.json"
            archive = Path(folder) / "result.json.gz"
            plain.write_bytes(raw)
            archive.write_bytes(gzip.compress(raw, mtime=0))
            self.assertEqual(gzip.decompress(archive.read_bytes()), raw)
            self.assertEqual(a.read_payload(plain), evidence)
            self.assertEqual(a.read_payload(archive), a.read_payload(plain))

    def test_corrupt_gzip_is_rejected_without_plain_json_fallback(self):
        with tempfile.TemporaryDirectory(prefix="experiment8-reader-test-") as folder:
            archive = Path(folder) / "broken.json.gz"
            archive.write_text('{"status":"pass"}', encoding="utf-8")
            with self.assertRaises(gzip.BadGzipFile):
                a.read_payload(archive)


class OfflineGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Isolate gate arithmetic from the frozen, explicitly incomplete real
        # task evaluator. Synthetic evidence uses a deterministic toy predicate.
        cls.real_evaluator = staticmethod(a.evaluate_quality_check)
        cls.evaluator_patch = patch.object(a, "evaluate_quality_check", return_value={"synthetic_fact": True})
        cls.evaluator_patch.start()
        cls.correctness = {arm:payload(arm,"correctness") for arm in a.ARMS}
        cls.screen = [payload(arm,multiplier=1 if arm=="C" else .94) for arm in a.ARMS]
        cls.full = [(payload("C","full",slot=0),payload("Q","full",.89,slot=1)),
                    (payload("C","full",slot=3),payload("Q","full",.89,slot=2))]

    @classmethod
    def tearDownClass(cls):
        cls.evaluator_patch.stop()

    def test_current_unmeasured_quality_scope_blocks_promotion(self):
        evidence=copy.deepcopy(self.correctness)
        evidence["C"]["absolute_quality_scope"]="Declared structural/fact checks only; no extraction schema or semantic evaluation is claimed."
        result=a.analyze_confirmation(self.full,"Q",evidence)
        self.assertEqual(result["outcome"],"inconclusive")
        self.assertTrue(result["absolute_quality_limitation"])

    def test_quality_raw_text_is_recomputed(self):
        evidence=copy.deepcopy(self.correctness)
        next(iter(evidence["Q"]["checks"].values()))["quality"]={"synthetic_fact": False}
        self.assertTrue(a.checked(evidence["Q"],"correctness"))

    def test_complete_raw_evidence_selects_one_deterministic_finalist(self):
        result = a.analyze_screen(self.screen,self.correctness)
        self.assertEqual(result["finalist"],"Q")  # Same gain: one family, then fixed rank.

    def test_larger_gain_precedes_fewer_families(self):
        screens=copy.deepcopy(self.screen)
        for case in screens[3]["cases"]:
            for run in [case["warmup"],*case["runs"]]: run["request_seconds"] = .90
        self.assertEqual(a.analyze_screen(screens,self.correctness)["finalist"],"QG")

    def test_missing_prerequisites_or_arms_cannot_select(self):
        self.assertIsNone(a.analyze_screen(self.screen)["finalist"])
        self.assertIsNone(a.analyze_screen(self.screen[:2],self.correctness)["finalist"])

    def test_control_quality_failure_blocks_even_fast_candidates(self):
        evidence=copy.deepcopy(self.correctness)
        evidence["C"]["absolute_quality_pass"] = False
        result=a.analyze_screen(self.screen,evidence)
        self.assertIsNone(result["finalist"])
        self.assertTrue(result["candidates"][0]["mechanism_screen_pass"])

    def test_contradictory_raw_quality_blocks_false_pass_flag(self):
        evidence=copy.deepcopy(self.correctness)
        next(iter(evidence["C"]["checks"].values()))["quality"]["synthetic_fact"] = False
        self.assertIsNone(a.analyze_screen(self.screen,evidence)["finalist"])

    def test_summary_and_pass_status_cannot_hide_missing_runs(self):
        broken=copy.deepcopy(self.screen[1]); broken["cases"][0]["runs"].pop()
        result=a.compare(self.screen[0],broken,"screen",self.correctness)
        self.assertFalse(result["eligible"]); self.assertTrue(result["errors"])

    def test_duplicate_cases_not_counted_as_complete(self):
        broken=copy.deepcopy(self.screen[1]); broken["cases"][-1] = copy.deepcopy(broken["cases"][0])
        self.assertTrue(a.checked(broken,"screen"))

    def test_token_cache_and_prompt_mismatch_rejected(self):
        for field in ("token","cache","prompt"):
            with self.subTest(field=field):
                broken=copy.deepcopy(self.screen[1]); case=broken["cases"][0]
                if field=="prompt": case["prompt_sha256"]="c"*64
                else:
                    for record in [case["warmup"],*case["runs"]]:
                        if field=="token": record["output_token_ids"][0]=123
                        else: record["cache"][0]["bytes"]=9999
                self.assertFalse(a.compare(self.screen[0],broken,"screen",self.correctness)["eligible"])

    def test_swap_growth_and_unmeasured_rss_reject(self):
        broken=copy.deepcopy(self.screen[1]); broken["state_end"]["swap_used_bytes"]+=1
        self.assertTrue(a.checked(broken,"screen"))
        broken["state_end"]["swap_used_bytes"]-=1
        broken["state_start"]["process_rss_bytes"]=None
        self.assertTrue(a.checked(broken,"screen"))

    def test_numerical_grid_missing_point_blocks(self):
        evidence=copy.deepcopy(self.correctness); evidence["Q"]["packing"][0]["verification"]["checks"].pop()
        self.assertFalse(a.compare(self.screen[0],self.screen[1],"screen",evidence)["eligible"])

    def test_nan_metric_and_wrong_peak_units_reject(self):
        broken=copy.deepcopy(self.screen[1]); broken["cases"][0]["runs"][0]["request_seconds"]=float("nan")
        self.assertTrue(a.checked(broken,"screen"))
        broken["cases"][0]["runs"][0]["request_seconds"]=1.
        broken["cases"][0]["runs"][0]["memory"]["peak_bytes"]=1
        self.assertTrue(a.checked(broken,"screen"))

    def test_request_memory_and_startup_peak_are_independent_gates(self):
        broken=copy.deepcopy(self.screen[1]); broken["memory_packing"]["peak_bytes"]+=513*2**20
        self.assertFalse(a.compare(self.screen[0],broken,"screen",self.correctness)["eligible"])
        broken=copy.deepcopy(self.screen[1])
        for case in broken["cases"]:
            for run in case["runs"]:
                run["mlx_peak_memory_gb"]=1.051; run["memory"]["peak_bytes"]=1.051e9
        self.assertFalse(a.compare(self.screen[0],broken,"screen",self.correctness)["eligible"])

    def test_full_pairs_pass_independently_and_reversed_order(self):
        result=a.analyze_confirmation(self.full,"Q",self.correctness)
        self.assertEqual(result["outcome"],"select")
        self.assertFalse(result["repeat_results_combined"])

    def test_failed_repeat_is_not_averaged_away(self):
        pairs=copy.deepcopy(self.full)
        for case in pairs[1][1]["cases"]:
            if case["target_prompt_tokens"] == 64:
                for run in case["runs"]: run["request_seconds"] = .91
        self.assertEqual(a.analyze_confirmation(pairs,"Q",self.correctness)["outcome"],"inconclusive")

    def test_full_p95_and_per_fixture_gates_use_raw_runs(self):
        pair=copy.deepcopy(self.full[0]); candidate=pair[1]
        # Six / fifty outliers cause the nearest-rank 95th percentile to regress,
        # while every fixture retains the same passing median.
        short=[c for c in candidate["cases"] if c["target_prompt_tokens"]==64]
        for case in short[:2]:
            for run in case["runs"][:3]: run["request_seconds"]=1.11
        result=a.analyze_pair(*pair,self.correctness)
        self.assertTrue(result["gates"]["both_short_request_improvement_at_least_10_percent"])
        self.assertFalse(result["gates"]["all_lengths_p95_request_and_ttft_regression_at_most_10_percent"])
        self.assertFalse(result["eligible"])

    def test_reused_or_reordered_capture_cannot_confirm(self):
        self.assertEqual(a.analyze_confirmation([self.full[0],self.full[0]],"Q",self.correctness)["outcome"],"inconclusive")
        self.assertEqual(a.analyze_confirmation(self.full,"G",self.correctness)["outcome"],"inconclusive")

    def test_stage0_manifest_drift_invalidates_timing(self):
        broken=copy.deepcopy(self.screen[1]); broken["manifest"]["benchmarks/experiment8_packing.py"]="c"*64
        self.assertFalse(a.compare(self.screen[0],broken,"screen",self.correctness)["eligible"])

    def test_packed_byte_flags_cannot_hide_bad_shape_accounting(self):
        evidence=copy.deepcopy(self.correctness)
        evidence["Q"]["packing"][0]["partitions"][0]["components"]["weight"]["source"]["shape"][1]=255
        self.assertFalse(a.compare(self.screen[0],self.screen[1],"screen",evidence)["eligible"])

    def test_fixture_order_must_match_control(self):
        broken=copy.deepcopy(self.screen[1]); broken["cases"].reverse()
        self.assertFalse(a.compare(self.screen[0],broken,"screen",self.correctness)["eligible"])

    def test_malformed_numeric_evidence_fails_closed(self):
        broken=copy.deepcopy(self.screen[1]); broken["packing"][0]["components"]["weight"]["dtype"]=123
        self.assertTrue(a.checked(broken,"screen"))

    def test_nearest_rank_uses_upper_tail_not_interpolation(self):
        self.assertEqual(a.nearest_rank(list(range(1,51))),48)
        self.assertEqual(a.nearest_rank(list(range(1,11))),10)


if __name__ == "__main__":
    unittest.main()

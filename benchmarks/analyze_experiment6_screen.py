#!/usr/bin/env python3
"""Compare separate-process Experiment 6 Stage-1 captures."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_complete(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("experiment") != "experiment-6-speculative-decoding-stage-1":
        raise ValueError(f"Not an Experiment 6 Stage-1 result: {path}")
    if "completed_at_utc" not in payload:
        raise ValueError(f"Incomplete result: {path}")
    for case in payload["cases"]:
        expected = payload["settings"]["measured_runs_per_fixture"]
        if len(case["runs"]) != expected:
            raise ValueError(f"Incomplete case {case['id']} in {path}")
    return payload


def label(payload: dict) -> str:
    return payload["arm"] if payload["arm"] != "draft" else f"draft-{payload['draft_count']}"


def change_percent(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def flattened_runs(payload: dict, max_tokens: int | None = None) -> list[dict]:
    return [
        run
        for case in payload["cases"]
        if max_tokens is None or case["max_tokens"] == max_tokens
        for run in case["runs"]
    ]


def metric_median(payload: dict, metric: str, max_tokens: int | None = None) -> float:
    return statistics.median(run[metric] for run in flattened_runs(payload, max_tokens))


def compare_exact(control: dict, candidate: dict) -> dict:
    control_cases = {case["id"]: case for case in control["cases"]}
    candidate_cases = {case["id"]: case for case in candidate["cases"]}
    if set(control_cases) != set(candidate_cases):
        raise ValueError(f"Case sets differ for {label(candidate)}")
    comparisons = []
    for case_id, control_case in control_cases.items():
        candidate_case = candidate_cases[case_id]
        if control_case["prompt_token_ids_sha256"] != candidate_case["prompt_token_ids_sha256"]:
            raise ValueError(f"Prompt mismatch for {case_id} in {label(candidate)}")
        control_outputs = [control_case["warmup"], *control_case["runs"]]
        candidate_outputs = [candidate_case["warmup"], *candidate_case["runs"]]
        reference_tokens = control_outputs[0]["output_token_ids"]
        control_deterministic = all(
            output["output_token_ids"] == reference_tokens for output in control_outputs
        )
        candidate_matches = all(
            output["output_token_ids"] == reference_tokens for output in candidate_outputs
        )
        comparisons.append({
            "case_id": case_id,
            "control_deterministic": control_deterministic,
            "candidate_exact_match": candidate_matches,
        })
    return {
        "case_count": len(comparisons),
        "control_deterministic": all(item["control_deterministic"] for item in comparisons),
        "candidate_exact_match": all(item["candidate_exact_match"] for item in comparisons),
        "cases": comparisons,
    }


def grouped_changes(control: dict, candidate: dict) -> dict:
    grouped = {}
    for target in control["settings"]["targets"]:
        grouped[str(target)] = {}
        for cap in (32, 128):
            control_runs = [
                run
                for case in control["cases"]
                if case["target_prompt_tokens"] == target and case["max_tokens"] == cap
                for run in case["runs"]
            ]
            candidate_runs = [
                run
                for case in candidate["cases"]
                if case["target_prompt_tokens"] == target and case["max_tokens"] == cap
                for run in case["runs"]
            ]
            grouped[str(target)][str(cap)] = {
                metric: change_percent(
                    statistics.median(run[metric] for run in candidate_runs),
                    statistics.median(run[metric] for run in control_runs),
                )
                for metric in (
                    "time_to_first_token_seconds",
                    "request_seconds",
                    "decode_tokens_per_second",
                    "mlx_peak_memory_gb",
                )
            }
    return grouped


def summarize_candidate(control: dict, candidate: dict) -> dict:
    exact = compare_exact(control, candidate)
    grouped = grouped_changes(control, candidate)
    summary = {
        "arm": label(candidate),
        "exactness": exact,
        "changes_percent": {},
        "by_target_and_cap_percent": grouped,
        "cold_process_to_first_token_change_percent": change_percent(
            candidate["cold_process_to_first_token_seconds"],
            control["cold_process_to_first_token_seconds"],
        ),
        "active_memory_after_load": {
            "control_gb": control["memory_after_load"]["target_only_active_gb"],
            "candidate_gb": candidate["memory_after_load"]["target_and_draft_active_gb"],
        },
    }
    for cap in (32, 128):
        summary["changes_percent"][str(cap)] = {
            metric: change_percent(
                metric_median(candidate, metric, cap), metric_median(control, metric, cap)
            )
            for metric in (
                "time_to_first_token_seconds",
                "request_seconds",
                "decode_tokens_per_second",
                "mlx_peak_memory_gb",
            )
        }
    draft_runs = [run for run in flattened_runs(candidate) if run.get("draft_metrics")]
    if draft_runs:
        summary["draft_metrics_median"] = {
            metric: statistics.median(run["draft_metrics"][metric] for run in draft_runs)
            for metric in (
                "draft_origin_fraction",
                "reconstructed_acceptance_fraction",
                "accepted_per_verification_round",
            )
        }
    long_changes = summary["changes_percent"]["128"]
    worst_peak_change = max(
        cap_values["mlx_peak_memory_gb"]
        for target_values in grouped.values()
        for cap_values in target_values.values()
    )
    control_cases = {case["id"]: case for case in control["cases"]}
    worst_absolute_peak_increase = max(
        statistics.median(run["mlx_peak_memory_gb"] for run in case["runs"])
        - statistics.median(
            run["mlx_peak_memory_gb"] for run in control_cases[case["id"]]["runs"]
        )
        for case in candidate["cases"]
    )
    maximum_candidate_peak = max(
        run["mlx_peak_memory_gb"] for run in flattened_runs(candidate)
    )
    summary["stage1_gate"] = {
        "exact_tokens": exact["candidate_exact_match"] and exact["control_deterministic"],
        "decode_improvement_at_least_5_percent": long_changes["decode_tokens_per_second"] >= 5.0,
        "request_regression_no_more_than_10_percent": long_changes["request_seconds"] <= 10.0,
        "peak_memory_no_more_than_50_percent_higher": worst_peak_change <= 50.0,
        "peak_memory_increase_no_more_than_0_8_gb": worst_absolute_peak_increase <= 0.8,
        "peak_memory_below_3_gb": maximum_candidate_peak < 3.0,
    }
    summary["stage1_gate"]["passed"] = all(summary["stage1_gate"].values())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    args = parser.parse_args()

    control = load_complete(args.control)
    if control["arm"] != "target":
        raise ValueError("--control must be the target-only arm")
    candidates = [load_complete(path) for path in args.candidate]
    output = {
        "control": str(args.control),
        "candidates": [summarize_candidate(control, candidate) for candidate in candidates],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

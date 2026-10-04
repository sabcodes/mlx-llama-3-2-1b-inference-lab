#!/usr/bin/env python3
"""Run Experiment 4's deterministic practical-quality gate on one MLX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "benchmarks" / "fixtures" / "experiment4-quality.json"
DISTRACTOR_SENTENCES = (
    "The north desk holds pencils and blank paper.",
    "A green folder contains last month's maintenance notes.",
    "The afternoon meeting begins after the lunch break.",
    "Window plants are watered every Wednesday.",
    "The storage room has four labeled shelves.",
    "Visitors sign the paper log beside the entrance.",
    "A wall clock hangs above the reception counter.",
    "The south hallway leads to two quiet rooms.",
    "Staff replace the notice board items each month.",
    "A blue binder lists ordinary supply requests.",
)


def normalize_exact(text: str) -> str:
    return re.sub(r"[\s.!?]+$", "", text.strip()).upper()


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else stripped


def check_output(text: str, checks: dict) -> dict:
    lowered = " ".join(text.lower().split())
    details: list[dict] = []

    for alternatives in checks.get("required_phrase_groups", []):
        passed = any(phrase.lower() in lowered for phrase in alternatives)
        details.append({"check": "required_phrase_group", "expected": alternatives, "passed": passed})

    for phrase in checks.get("forbidden_phrases", []):
        passed = phrase.lower() not in lowered
        details.append({"check": "forbidden_phrase", "expected_absent": phrase, "passed": passed})

    if expected := checks.get("exact_normalized"):
        actual = normalize_exact(text)
        details.append({"check": "exact_normalized", "expected": expected, "actual": actual, "passed": actual == expected})

    parsed_json = None
    if checks.get("valid_json"):
        try:
            parsed_json = json.loads(strip_json_fence(text))
            passed = True
            error = None
        except json.JSONDecodeError as exception:
            passed = False
            error = str(exception)
        details.append({"check": "valid_json", "passed": passed, "error": error})

    if required_strings := checks.get("json_required_strings"):
        serialized = json.dumps(parsed_json, ensure_ascii=False).lower() if parsed_json is not None else ""
        for expected in required_strings:
            details.append({
                "check": "json_required_string",
                "expected": expected,
                "passed": expected.lower() in serialized,
            })

    if expected_count := checks.get("bullet_count"):
        actual_count = len(re.findall(r"^\s*[-*•]\s+", text, re.MULTILINE))
        details.append({
            "check": "bullet_count",
            "expected": expected_count,
            "actual": actual_count,
            "passed": actual_count == expected_count,
        })

    return {"passed": all(item["passed"] for item in details), "checks": details}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    args = parser.parse_args()

    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    model_path = model_path.resolve()
    try:
        relative_model_path = model_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exception:
        raise ValueError("Model path must remain inside the project") from exception

    fixture_path = args.fixtures if args.fixtures.is_absolute() else ROOT / args.fixtures
    fixture_payload = json.loads(fixture_path.read_text())
    fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()

    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(model_path))
    sampler = make_sampler(temp=0.0)
    cases = []
    distractors = " ".join(DISTRACTOR_SENTENCES * 14)

    for fixture in fixture_payload["cases"]:
        user_text = fixture["user"].replace("{{DISTRACTORS}}", distractors)
        messages = [
            {"role": "system", "content": fixture["system"]},
            {"role": "user", "content": user_text},
        ]
        token_ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        responses = list(stream_generate(
            model,
            tokenizer,
            token_ids,
            max_tokens=fixture["max_tokens"],
            sampler=sampler,
            prefill_step_size=2048,
        ))
        if not responses:
            raise RuntimeError(f"No generation response for {fixture['id']}")
        output = "".join(response.text for response in responses)
        final = responses[-1]
        evaluation = check_output(output, fixture["checks"])
        cases.append({
            "id": fixture["id"],
            "prompt_tokens": final.prompt_tokens,
            "generated_tokens": final.generation_tokens,
            "finish_reason": final.finish_reason,
            "output": output,
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "evaluation": evaluation,
        })
        print(f"{fixture['id']}: {'PASS' if evaluation['passed'] else 'FAIL'}", flush=True)

    passed_cases = sum(case["evaluation"]["passed"] for case in cases)
    result = {
        "schema_version": 1,
        "experiment": "experiment-4-weight-quantization-quality-gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"id": args.model_id, "local_path": relative_model_path},
        "fixture": {
            "path": fixture_path.resolve().relative_to(ROOT.resolve()).as_posix(),
            "sha256": fixture_sha256,
            "case_count": len(cases),
        },
        "settings": {"sampling": "greedy", "prefill_step_size": 2048, "concurrency": 1},
        "summary": {
            "passed_cases": passed_cases,
            "failed_cases": len(cases) - passed_cases,
            "pass_rate": passed_cases / len(cases),
            "passed": passed_cases == len(cases),
        },
        "cases": cases,
    }
    write_json(args.output, result)
    print(f"Saved {args.output}: {passed_cases}/{len(cases)} cases passed", flush=True)


if __name__ == "__main__":
    main()

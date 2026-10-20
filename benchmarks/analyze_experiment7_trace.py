#!/usr/bin/env python3
"""Export a saved Instruments pilot to sanitized, overlap-aware shader evidence."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def union_ns(intervals):
    end = total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def rows(xml):
    root = ET.fromstring(xml)
    ids = {e.get("id"): e for e in root.iter() if e.get("id")}

    def resolve(e):
        while e.get("ref"):
            e = ids[e.get("ref")]
        return e

    columns = [c.findtext("mnemonic") for c in root.findall(".//schema/col")]
    return [{k: resolve(v) for k, v in zip(columns, row)}
            for row in root.findall(".//row")]


def family(name):
    if name.startswith(("affine_qmm", "affine_qmv", "affine_dequantize")):
        return "quantized_matrix_and_dequantization"
    if name.startswith(("sdpa_", "steel_attention")):
        return "attention"
    if name.startswith("rms"):
        return "normalization"
    if any(s in name for s in ("copy", "gather", "scatter")):
        return "copy_and_indexing_unassigned"
    return "elementwise_sampling_and_other"


def summarize(xml, pid):
    all_rows = rows(xml)
    target = [r for r in all_rows if r["process"].findtext("pid") == pid]
    groups = defaultdict(list)
    labels = defaultdict(list)
    for r in target:
        start, duration = int(r["start"].text), int(r["duration"].text)
        assert start >= 0 and duration >= 0
        name = r["name"].text
        groups[family(name)].append((start, start + duration))
        labels[name].append((start, start + duration))
    intervals = [i for group in groups.values() for i in group]
    summed = sum(b - a for a, b in intervals)
    assert summed > 0, "No target shader durations; attribution is unavailable"
    return {
        "total_rows": len(all_rows), "target_rows": len(target),
        "excluded_rows": len(all_rows) - len(target),
        "target_depth_counts": dict(Counter(r["event-depth"].text for r in target)),
        "first_target_sample_ns": min(a for a, b in intervals),
        "last_target_sample_end_ns": max(b for a, b in intervals),
        "summed_sample_duration_ns": summed,
        "union_sample_coverage_ns": union_ns(intervals),
        "families": {k: {"rows": len(v),
                          "summed_sample_duration_ns": sum(b - a for a, b in v),
                          "share_of_summed_target_sample_duration": sum(b - a for a, b in v) / summed,
                          "union_sample_coverage_ns": union_ns(v)}
                     for k, v in sorted(groups.items())},
        "shaders": {k: {"rows": len(v), "summed_sample_duration_ns": sum(b - a for a, b in v)}
                    for k, v in sorted(labels.items())},
    }


def self_test():
    assert union_ns([(0, 10), (5, 20), (30, 40)]) == 30
    assert union_ns([]) == 0
    xml = '''<root><schema><col><mnemonic>process</mnemonic></col></schema>
    <row><process id="1"><pid>42</pid></process></row>
    <row><process ref="1"/></row></root>'''
    assert [r["process"].findtext("pid") for r in rows(xml)] == ["42", "42"]
    assert family("affine_qmv_fast") == "quantized_matrix_and_dequantization"
    assert family("vn_copyfloat16") == "copy_and_indexing_unassigned"
    xml = """<root><schema>
    <col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col>
    <col><mnemonic>name</mnemonic></col><col><mnemonic>process</mnemonic></col>
    <col><mnemonic>event-depth</mnemonic></col></schema>
    <row><start-time>0</start-time><duration>10</duration>
    <label id="2">affine_qmm_test</label><process id="1"><pid>42</pid></process><int>0</int></row>
    <row><start-time>5</start-time><duration>10</duration>
    <label ref="2"/><process ref="1"/><int>1</int></row>
    <row><start-time>0</start-time><duration>1000</duration>
    <label>unrelated_shader</label><process><pid>99</pid></process><int>0</int></row></root>"""
    result = summarize(xml, "42")
    assert result["target_rows"] == 2 and result["excluded_rows"] == 1
    assert result["summed_sample_duration_ns"] == 20
    assert result["union_sample_coverage_ns"] == 15
    print("Reference resolution, target isolation, overlap union, and label classification checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.trace or not args.output:
        parser.error("--trace and --output are required")
    trace, output = args.trace.resolve(), args.output.resolve()
    if not trace.is_relative_to(ROOT) or not output.is_relative_to(ROOT) or output.exists():
        parser.error("Use an existing project trace and a new project output path")

    def export(*options):
        return subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace), *options],
                              check=True, capture_output=True, timeout=60).stdout

    toc_xml = export("--toc")
    toc = ET.fromstring(toc_xml)
    process = toc.find(".//target/process")
    assert process.get("name") == "Python", "Expected a Python target"
    xml = export("--xpath", '/trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-intervals"]')
    result = {
        "schema_version": 1, "experiment": "experiment-7-m2pro-cli-pilot-attribution",
        "trace_name": trace.name,
        "trace_logical_bytes": sum(p.stat().st_size for p in trace.rglob("*") if p.is_file() and not p.is_symlink()),
        "trace_readable": True, "tool": toc.findtext(".//instruments-version"),
        "recording_duration_seconds": float(toc.findtext(".//summary/duration")),
        "shader_xml_sha256": hashlib.sha256(xml).hexdigest(),
        "toc_xml_sha256": hashlib.sha256(toc_xml).hexdigest(),
        "metric": "metal-shader-profiler-intervals Sample Duration; raw integer nanoseconds",
        "denominator": "Sum of Sample Duration over all rows for the attached Python target",
        "overlap_treatment": "Sum retains overlapping samples; union coverage is also reported. Family unions are not additive.",
        "phase_specific_attribution_complete": False,
        "limitations": [
            "Mixed prefill/decode window, not separate phase captures; third request extends beyond the recording.",
            "Sample-duration shares are not exact dispatch durations, clean request-time shares, or M4 replay shader costs.",
            "Copy/indexing labels do not identify KV updates independently; KV update cost remains unknown.",
            "Unobserved work and unsampled intervals remain unknown, not zero.",
        ],
        **summarize(xml, process.get("pid")),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("target_rows", "summed_sample_duration_ns", "union_sample_coverage_ns", "families")}, indent=2))


if __name__ == "__main__":
    main()

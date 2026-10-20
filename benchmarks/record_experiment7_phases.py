#!/usr/bin/env python3
"""Record three bounded phase/KV diagnostics after warmup, with idle drain time."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks/results"
WORKER = ROOT / "benchmarks/run_experiment7_phase_worker.py"


def size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()) if path.exists() else 0


def combined_size():
    return sum(size(p) for p in BASE.glob("experiment7-m2pro-phase-*.trace"))


def record(phase, run_id):
    tag = f"experiment7-m2pro-phase-{phase}-{run_id}"
    trace = BASE / (tag + ".trace")
    ready = BASE / (tag + "-ready.partial.json")
    progress = BASE / (tag + "-progress.partial.json")
    output = BASE / (tag + "-workload.json")
    provenance = BASE / (tag + "-provenance.json")
    assert not any(p.exists() for p in (trace, ready, progress, output, provenance))
    worker = listener = recorder = None
    data = {"schema_version": 1, "phase": phase, "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "worker_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest(),
            "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "collector_settling_seconds": 2, "collector_idle_drain_seconds": 2,
            "additional_retained_limit_bytes": 1_000_000_000, "observed_working_limit_bytes": 2_000_000_000}
    peak = combined_size()
    try:
        worker = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-B", str(WORKER), "--phase", phase,
                                   "--ready", str(ready), "--progress", str(progress), "--output", str(output)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        deadline = time.monotonic()+90
        while not ready.exists():
            if worker.poll() is not None:
                raise RuntimeError(worker.communicate()[0])
            if time.monotonic() > deadline:
                raise TimeoutError("Worker readiness timeout")
            time.sleep(.1)
        assert json.loads(ready.read_text())["warmup_completed"]
        notification = f"com.sabcodes.experiment7.phase.{phase}.{os.getpid()}"
        listener = subprocess.Popen(["notifyutil", "-1", notification], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = ["xcrun", "xctrace", "record", "--template", "Metal System Trace", "--instrument", "Metal GPU Counters",
                   "--time-limit", "15s", "--output", str(trace), "--notify-tracing-started", notification, "--attach", str(worker.pid)]
        recorder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        began = time.monotonic()
        notified = completed = None
        triggered = stopped = False
        print(f"Recording {phase}; collector settling before trigger.", flush=True)
        while recorder.poll() is None:
            now = time.monotonic()
            if notified is None and listener.poll() is not None:
                assert listener.returncode == 0
                notified = now
            if not triggered and notified is not None and now-notified >= 2:
                os.kill(worker.pid, signal.SIGUSR1)
                triggered = True
            if completed is None and progress.exists():
                assert json.loads(progress.read_text())["recorded_work_completed"]
                completed = now
            if not stopped and completed is not None and now-completed >= 2:
                os.killpg(recorder.pid, signal.SIGINT)
                stopped = True
            peak = max(peak, combined_size())
            if peak > 2_000_000_000:
                raise RuntimeError("Observed working-output limit exceeded")
            if now-began > 120:
                raise TimeoutError("Recording finalization timeout")
            time.sleep(.2)
        log = recorder.communicate()[0]
        assert recorder.returncode == 0 and stopped, log[-1500:]
        assert combined_size() <= 1_000_000_000
        os.kill(worker.pid, signal.SIGUSR2)
        worker_log = worker.communicate(timeout=45)[0]
        assert worker.returncode == 0, worker_log[-2000:]
        result = json.loads(output.read_text())
        assert result.get("completed_at_utc")
        assert result.get("output_matches_clean_m2") if phase != "kv" else result.get("kv_numerical_checks_passed")
        data.update({"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                     "command": [v.replace(str(ROOT), "[project]").replace(str(worker.pid), "[worker-pid]") for v in command],
                     "trace_path": trace.relative_to(ROOT).as_posix(), "trace_command_exit_code": recorder.returncode,
                     "worker_exit_code": worker.returncode, "warmup_outside_recording": True,
                     "stopped_after_synchronized_phase_and_idle_drain": True, "target_held_alive_while_saving": True,
                     "trace_logical_bytes": size(trace), "combined_new_trace_bytes": combined_size(),
                     "observed_peak_combined_new_trace_bytes": peak,
                     "sanitized_tool_output": log.replace(str(ROOT), "[project]").replace(str(Path.home()), "[home]").replace(str(worker.pid), "[worker-pid]")[-2500:]})
        provenance.write_text(json.dumps(data, indent=2)+"\n")
        print(f"{phase}: saved, output verified; {size(trace)} retained bytes.", flush=True)
    finally:
        for process in (recorder, worker):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        if listener is not None and listener.poll() is None:
            listener.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9-]+", args.run_id):
        parser.error("Use an alphanumeric/hyphen run ID")
    for folder in (ROOT / ".venv", ROOT / "models/Llama-3.2-1B-Instruct-4bit"):
        assert not any((Path(parent)/name).stat().st_flags & 0x40000000
                       for parent, dirs, files in os.walk(folder) for name in files), "iCloud placeholders remain"
    for phase in ("prefill", "decode", "kv"):
        record(phase, args.run_id)
    print("Combined retained new traces:", combined_size(), flush=True)


if __name__ == "__main__":
    main()

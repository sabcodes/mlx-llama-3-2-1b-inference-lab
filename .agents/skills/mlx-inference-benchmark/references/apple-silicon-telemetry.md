# Apple Silicon GPU and thermal telemetry

## What this Mac exposes

On the current Mac, `/usr/bin/powermetrics --help` lists `gpu_power` (GPU power/frequency information), `thermal` (thermal-pressure notifications), and `--show-process-gpu` (per-process GPU time). It requires superuser privileges. These are not automatically equivalent to a single whole-GPU utilization percentage. Inspect the actual output and preserve the metric names as reported; keep `gpu_utilization_percent` null unless the sampled output provides a validated percentage with a clear interval and definition.

To collect a raw diagnostic log, have the user open the macOS Terminal app in the project root and run:

```sh
sudo /usr/bin/powermetrics \
  --sample-rate 1000 \
  --sample-count -1 \
  --samplers tasks,gpu_power,thermal \
  --show-process-gpu \
  --show-plimits \
  > benchmarks/results/powermetrics-<same-run-id>.txt 2>&1
```

The user enters the administrator password into that macOS Terminal prompt; never request or handle the password. Pause until the user confirms sampling has begun. Then start the benchmark from the project environment in a separate shell. Stop the collector with Ctrl-C after the benchmark exits. Keep this raw telemetry log beside its matching JSON and use the same run ID to associate them. Confirm the flags with `/usr/bin/powermetrics --help` on each Mac/macOS version before use; available samplers and output fields can differ.

This collector samples system telemetry while the benchmark runs, so it can add overhead. Use an uninstrumented benchmark JSON for primary latency comparisons and a separate instrumented run for telemetry. Do not compare an instrumented run's latency directly with an uninstrumented run. `--show-process-gpu` may help attribute GPU time to the benchmark process, while `gpu_power` provides device-level power/frequency data; neither should be relabeled as utilization without confirming the actual fields and units.

For a visual, non-exported cross-check, Activity Monitor has Window > GPU History. Apple's guide describes it as a view of how hard the GPU has been working, not a numeric benchmark export: [View GPU activity in Activity Monitor](https://support.apple.com/guide/activity-monitor/view-gpu-activity-actm9329b315/mac).

## Temperature versus thermal state

The `thermal` sampler reports thermal pressure/state, not a temperature in degrees. Apple's thermal-state API is also categorical (`nominal`, `fair`, `serious`, `critical`) and is intended to indicate when software should reduce work; see [Responding to power notifications](https://developer.apple.com/documentation/xcode/responding-to-power-notifications).

The current `powermetrics --help` on this Mac does not list the older `smc` sampler, so a built-in, supported numeric die-temperature reading is not available through this command here. Do not infer an internal temperature from thermal state, GPU power, fan speed, or a temperature reported by an unvalidated third-party sensor utility. For this project, record thermal pressure and any power/frequency/clock-limit evidence available from `powermetrics`; treat raw internal temperature as unavailable unless the user chooses a sensor tool and its sensor labels and readings are independently validated for this Mac and OS.

Apple's Activity Monitor can visualize GPU history, and Xcode/Metal tools can collect detailed GPU counters for supported Metal workloads, but those profiling workflows are not equivalent to the lightweight, repeatable MLX LM benchmark. Use them only for a targeted follow-up investigation, not as a replacement for the suite's throughput and latency measurements.

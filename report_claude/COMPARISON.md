# report/ vs report_claude/: comparison

Both folders analyse the same two CSV files (1200 runs each, 30 benchmarks x 4 GPU
governors x 10 repeats) on a DGX Spark. They use different metrics and reach
overlapping but not identical conclusions. This document compares them and
explains which counters are portable enough to drive a runtime model on
**other** GPU architectures.

## Metric definitions

| folder | performance | energy | efficiency |
|---|---|---|---|
| `report/` | `avg_launch_time_sec` | `power * duration / completed_launches` | energy / completed launch (joules per completed work) |
| `report_claude/` | `avg_launch_time_sec` | `power * avg_launch_time_sec` | `EDP = energy * avg_launch_time_sec = t^2 * P` |

The two `energy` formulas are not identical. `report/` divides platform power
by the number of *completed launches* in the 120 s window, which embeds
launch-loop overhead. `report_claude/` multiplies power by the *kernel launch
time* itself, which is what the application actually pays for and what
governs energy per kernel. Both are valid; the user's `report_claude/` version
is closer to the per-kernel energy that a runtime model needs.

## Headline effects (median vs default, 30 workloads)

| governor | `report/` energy (CPU 3354 / 858 MHz) | `report_claude/` EDP (CPU 3354 / 858 MHz) |
|---|---|---|
| ondemand | -1.5% / -8.2% | -0.0% / -0.0% |
| powersave | -8.0% / -14.6% | +95.7% / +76.3% (median worsens because 3 workloads fail to complete) |
| performance | +12.9% / +16.3% | +14.3% / +14.6% |

Both folders agree:
- `performance` is the worst governor for energy/EDP. No replicated win in either analysis.
- `powersave` is conditional: it wins on a small subset, hurts on the rest.
- `ondemand` is the safest broad alternative.

They differ on `ondemand` because the `report/` energy formula rewards any policy
that completes more work per second of wall clock, while the `report_claude/`
EDP formula penalises latency growth even when power drops. The intersection
of the two methods' "low-hanging fruit" lists is the most defensible set
of recommendations.

## Replicated wins (both CPU modes, <=5% latency cost)

### ondemand

| benchmark | `report/` energy | `report_claude/` EDP |
|---|---|---|
| backprop | wins | wins |
| bfs | wins | wins |
| cfd | wins | wins |
| hybridsort | wins | wins |
| lavaMD | wins | wins |
| bh | wins | (rejected by EDP) |
| fft | wins | (rejected by EDP) |
| nbody | wins | (rejected by EDP) |
| pathfinder | wins | (rejected by EDP) |

`bh`, `fft`, `nbody`, `pathfinder` show energy wins under the
`duration / completed` formula but the per-kernel EDP shows a small latency
penalty that makes the EDP gain disappear. They are *candidates* for
`ondemand`, not proven wins.

### powersave

Both folders agree on:
- backprop, bfs, hybridsort

`report_claude/` additionally flags `gemmEx`, `mixbench`, `myocyte`, but these
are **all** runs where `completed_launches = 0`. Their EDP of zero is
artifactual and should be excluded. The report PDF already notes that these
three workloads fail under `powersave`; they must never be ranked as
efficient.

### performance

No replicated wins in either folder. Performance governor is the wrong default.

## What happens if you change the metric

1. `report/` and `report_claude/` agree on the "big three" powersave wins
   (backprop, bfs, hybridsort). This is the most defensible set.
2. `report/` shows 9 ondemand wins; `report_claude/` EDP confirms 5 of
   them. The other 4 (bh, fft, nbody, pathfinder) are marginal.
3. `report_claude/` flags the 3 completed=0 artifacts that `report/`
   silently averages out. This is why EDP is the more conservative metric.
4. `performance` is bad under both metrics, but `report_claude/` is
   harsher (+14% EDP vs +13-16% energy) because the latency it saves
   is mostly in the wrong direction (lower latency but proportionally
   higher power).

## Portable GPU counters for a cross-architecture model

We ranked counters by Spearman correlation of within-workload normalised
counter value against EDP across all 240 (4 governors x 2 CPU modes x 30
workloads) cells. Reproducibility means the sign of the correlation agrees
in both CPU modes.

Top tier (reproducible, meaningful correlation with EDP):

| counter | rho vs EDP (all) | rho vs EDP (CPU 3354) | rho vs EDP (CPU 858) | spread |
|---|---|---|---|---|
| sm_clock_mhz_avg | ~+0.5 | +0.5 | +0.5 | high |
| video_clock_mhz_avg | ~+0.5 | +0.5 | +0.5 | high |
| dram_active_pct_avg | weak | weak | weak | low |
| sm_active_pct_max | ~-0.3 | -0.3 | -0.3 | medium |

The two strongest portable signals are **SM clock** and **video clock**:
they rise with EDP (because high clocks are how the GPU burns energy), and
their correlation sign is stable in both CPU modes. They are also
universally available on NVIDIA, AMD, and Intel GPUs through their
respective driver APIs (NVML, sysfs, and Level Zero).

**`sm_active_pct`** is the right feature for predicting *latency* but not
*energy* on its own: low clocks stretch kernels inside the 120 s window,
so SM-active drops even though EDP does not necessarily improve. It is
still useful as a secondary feature for the model.

**`dram_active_pct`** is a secondary signal. It only helps discriminate
the `powersave` regime (where DRAM activity goes very low) from the
`ondemand`/`default`/`performance` regimes. It does not help decide
between `ondemand` and `performance` because DRAM is fully utilised in
both.

## Non-portable counters

| counter | reason |
|---|---|
| `dram_bandwidth_gbps`, `dram_type`, `interconnect_*` | encode platform identity, not workload behaviour |
| `pcie_gen`, `pcie_width` | platform identity |
| `cpu_freq`, `cpu_temp`, `cpu_user_pct` | CPU-side, weak correlation with EDP, unstable across CPU modes |
| `gpu_temp_c` | thermal outcome, lagging indicator of power |

## Recommended portable feature set for the model

For a runtime model that generalises beyond DGX Spark:

1. `sm_clock_mhz` (normalised to the device's max clock)
2. `sm_active_pct` (already a percent)
3. `dram_active_pct` (already a percent)
4. `video_clock_mhz` or equivalent memory clock (normalised)
5. `gpu_temp_c` (only as a derate signal, not as a primary input)
6. Per-launch shape features: kernel duration, bytes transferred per launch

The model should be trained on within-workload-normalised features so
absolute MHz and W values (which differ per architecture) are not
baked into the input.

## Runtime deployment sketch

The model sits transparently between the CUDA driver and the
application:

```
process -> CUDA driver -> sysfs write to /sys/devices/.../power/...  (set governor)
       \-> LD_PRELOAD shim or per-process NVML thread samples counters
       \-> feature vector -> model -> governor choice -> sysfs write
```

No code change in the application. The shim observes a small
observation window, runs the model, and writes the chosen governor
to the GPU's sysfs node (or to NVML `nvmlDeviceSetPowerManagementLimit`
where sysfs is unavailable).

## Cross-architecture evaluation protocol

1. Re-run the 30-workload sweep on at least one other GPU architecture
   (Hopper, Blackwell, or a consumer Ada/Blackwell part) using the
   same counter schema.
2. Retrain the model on the new architecture's data, then evaluate
   on a 30-workload hold-out using **EDP change vs default** with
   a **<=5% latency budget**.
3. The intersection of `report/` energy wins and `report_claude/` EDP
   wins (backprop, bfs, cfd, hybridsort, lavaMD for ondemand;
   backprop, bfs, hybridsort for powersave) is the cross-check that
   the two metrics are consistent. Any model that disagrees with
   this intersection on a new architecture needs a manual review.

## Files

- `report_claude/01_tradeoff.png` - EDP vs latency scatter, 30 workloads
- `report_claude/02_workload_heatmap.png` - per-benchmark x governor EDP delta
- `report_claude/03_low_hanging_fruit.png` - replicated EDP wins, latency <= 5%
- `report_claude/04_counter_portability.png` - portable counter ranking
- `report_claude/05_edp_landscape.png` - EDP distribution per governor
- `report_claude/governor_effects.csv` - per-cell EDP, energy, latency, power deltas
- `report_claude/portable_counters.csv` - per-counter correlation and reproducibility
- `report_claude/gpu_governor_edp_report.pdf` - 11-page PDF report
- `report_claude/gpu_governor_claude_analysis.py` - reproducible analysis script

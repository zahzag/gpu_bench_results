# Paired DGX Spark CPU/GPU Governor Analysis

## Scope and method

This report pairs **2,400 runs**, **240 cells**, **30 applications**, four GPU governors, and two CPU modes. CPU powersave was measured at 858 MHz in the experiment beginning 20260716_163946 (file timestamps 20260716_164154 to 20260718_190616); CPU performance was measured at 3354 MHz in experiment 20260720_191544 (file timestamps 20260720_191752 to 20260722_214012). Every cell has 10 runs.

The metric is strictly `GPU_EDP = avg_gpu_time_sec^2 * power_draw_w_avg`, calculated per run and then aggregated as the median of the 10 per-run EDP values. `avg_launch_time_sec` is never used in EDP; it is application latency and supplies the <=5% guardrail. CPU EDP is omitted because DGX Spark CPU package power is unavailable. CPU time is analyzed only as latency.

## Exact aggregate effects

Across all 120 matched CPU-mode/governor/application pairs, changing CPU powersave to performance changed median application latency by **-2.76%** (median). Across the 110 valid GPU-EDP pairs, it changed GPU EDP by **-0.17%** (median). Median CPU-time change was **+0.27%**. These paired summaries show that CPU governor effects on application latency and GPU EDP are distinct and must not be conflated.

Robust recommendations are portable only when the same GPU governor has valid EDP in both CPU modes and stays within +5% application latency in both. **2 of 30 applications** obtain a lower cross-CPU geometric-mean EDP than default. Recommendation counts are: Default 28, Ondemand 2. Top robust reductions are: backprop-cuda -51.49% via Ondemand, lavaMD-cuda -2.49% via Ondemand.

## Application recommendations

- **babelstream-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **backprop-cuda**: Ondemand; EDP -51.49%, powersave latency +1.89%, performance latency +2.12%; PASS.
- **bfs-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **bh-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **cfd-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **fft-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **gaussian-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **gemmEx-cuda**: Default; EDP NA, powersave latency +0.00%, performance latency +0.00%; UNRANKABLE.
- **heartwall-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **hotspot-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **hybridsort-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **ising-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **kmeans-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **lavaMD-cuda**: Ondemand; EDP -2.49%, powersave latency -0.20%, performance latency +0.00%; PASS.
- **leukocyte-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **lud-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **lulesh-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **memcpy-cuda**: Default; EDP NA, powersave latency +0.00%, performance latency +0.00%; UNRANKABLE.
- **mixbench-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **myocyte-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **nbody-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **nn-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **nw-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **particlefilter-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **pathfinder-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **srad-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **stencil1d-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **stencil3d-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **streamcluster-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.
- **triad-cuda**: Default; EDP +0.00%, powersave latency +0.00%, performance latency +0.00%; PASS.

## Censoring and instrumentation

Missing/`None` GPU time is a censored failure, never zero energy. The 6 censored cells are: cpu_performance:gemmEx-cuda:gpu_powersave, cpu_performance:mixbench-cuda:gpu_powersave, cpu_performance:myocyte-cuda:gpu_powersave, cpu_powersave:gemmEx-cuda:gpu_powersave, cpu_powersave:mixbench-cuda:gpu_powersave, cpu_powersave:myocyte-cuda:gpu_powersave. Any governor comparison involving these cells is unrankable.

Zero GPU time is an instrumentation limitation and is excluded from all EDP percentage rankings. Affected applications are **gemmEx-cuda, memcpy-cuda**. This explicitly includes `gemmEx-cuda` and `memcpy-cuda`; no zero-time observation is represented as a zero-energy win.

## Portable runtime policy

Use the static application-to-governor map in `recommendations.csv` only for rows marked `PASS`. At runtime, verify a positive GPU timer, completed launches, power sample count, GPU utilization, clocks, temperature/throttling state, problem size, and an application-latency sentinel. Fall back to `gpu_default` if GPU time is missing/zero, the workload is unknown, the latency sentinel exceeds +5%, or counters leave the calibration envelope. Periodically re-profile because power/thermal state and software versions can shift the optimum.

## Limitations

- The CPU-mode experiments occurred on different dates, so CPU mode is confounded with date, software state, thermal history, and other uncontrolled drift. Pairing by application/governor does not remove that confounding.
- CPU, GPU, memcpy, and launch timers may overlap and are nonadditive. The component plot is a bottleneck diagnostic, not a decomposition of launch latency.
- CPU package power is absent on DGX Spark, so CPU EDP and whole-system EDP cannot be computed. CPU time and launch time are latency metrics only.
- Median cell estimates summarize 10 runs but do not establish statistical significance or causal effects.
- Results cover the measured large input and this hardware/software environment; portability requires runtime validation.

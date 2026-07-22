#!/usr/bin/env python3
"""GPU governor analysis: report_claude.

This is an independent re-analysis of the same DGX Spark benchmark data using
the user-defined metrics:

- Performance metric: avg_launch_time_sec (latency per launch, lower = better)
- Energy metric: power_draw_w_avg (average platform power, lower = better)
- Energy per launch: avg_launch_time_sec * power_draw_w_avg (joules per launch)
- EDP (Energy-Delay Product): energy * avg_launch_time_sec = t^2 * P

The script also identifies which GPU counters are portable for a
cross-architecture governor-selection model, and produces:
- 01_tradeoff.png: EDP vs latency scatter
- 02_workload_heatmap.png: EDP delta per workload x governor
- 03_counter_portability.png: rank order of portable GPU counters
- 04_edp_landscape.png: EDP landscape across governors
- governor_effects.csv: per-benchmark governor effects
- portable_counters.csv: portable counter ranking
- gpu_governor_claude_report.pdf: multi-page PDF
"""

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "default": "#4C78A8",
    "ondemand": "#2A9D8F",
    "powersave": "#E9C46A",
    "performance": "#E76F51",
    "ink": "#17202A",
    "muted": "#667085",
    "grid": "#D9E1E8",
    "paper": "#F8FAFC",
    "good": "#167A5B",
    "bad": "#B5473C",
}


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def median(values):
    values = [v for v in values if math.isfinite(v)]
    return statistics.median(values) if values else math.nan


def pct(value, baseline):
    if not baseline or not math.isfinite(value) or not math.isfinite(baseline):
        return math.nan
    return 100.0 * (value / baseline - 1.0)


def spearman(xs, ys):
    if len(xs) < 3 or len(ys) < 3:
        return math.nan
    n = len(xs)
    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return math.nan
    return num / (dx * dy)


def rank(values):
    indexed = sorted(enumerate(values), key=lambda kv: (kv[1], kv[0]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def font(size, bold=False):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size)


def canvas(width=1800, height=1050):
    return Image.new("RGB", (width, height), COLORS["paper"])


def title(draw, text, subtitle=None):
    draw.text((80, 52), text, font=font(38, True), fill=COLORS["ink"])
    if subtitle:
        draw.text((82, 103), subtitle, font=font(20), fill=COLORS["muted"])


def load(path, cpu_mode):
    rows = []
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            governor = row["governor"].removeprefix("gpu_")
            benchmark = row["benchmark"].removesuffix("-cuda")
            power = fnum(row.get("power_draw_w_avg"))
            latency = fnum(row.get("avg_launch_time_sec"))
            energy = power * latency
            edp = energy * latency
            rows.append({
                "cpu_mode": cpu_mode,
                "governor": governor,
                "benchmark": benchmark,
                "run": int(row["run"]),
                "power": power,
                "latency": latency,
                "energy": energy,
                "edp": edp,
                "sm_clock": fnum(row.get("sm_clock_mhz_avg")),
                "sm_clock_max": fnum(row.get("sm_clock_mhz_max")),
                "sm_clock_min": fnum(row.get("sm_clock_mhz_min")),
                "video_clock": fnum(row.get("video_clock_mhz_avg")),
                "mem_clock": fnum(row.get("mem_clock_mhz_est")),
                "sm_active": fnum(row.get("sm_active_pct_avg")),
                "sm_active_max": fnum(row.get("sm_active_pct_max")),
                "sm_active_min": fnum(row.get("sm_active_pct_min")),
                "dram_active": fnum(row.get("dram_active_pct_avg")),
                "gpu_temp": fnum(row.get("gpu_temp_c_avg")),
                "cpu_temp": fnum(row.get("cpu_temp_c_avg")),
                "cpu_freq": fnum(row.get("cpu_freq_mhz_avg")),
                "cpu_user": fnum(row.get("cpu_user_pct_avg")),
                "completed": fnum(row.get("completed_launches")),
                "duration": fnum(row.get("duration_actual_sec")),
                "pcie_gen": fnum(row.get("pcie_gen_avg")),
                "pcie_width": fnum(row.get("pcie_width_avg")),
            })
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["cpu_mode"], r["benchmark"], r["governor"])].append(r)
    cells = {}
    for key, group in grouped.items():
        def m(name):
            return median([r[name] for r in group])
        cells[key] = {name: m(name) for name in (
            "power", "latency", "energy", "edp", "sm_clock", "sm_clock_max", "sm_clock_min",
            "video_clock", "mem_clock", "sm_active", "sm_active_max", "sm_active_min",
            "dram_active", "gpu_temp", "cpu_temp", "cpu_freq", "cpu_user",
            "completed", "duration", "pcie_gen", "pcie_width"
        )}
        cells[key]["valid_runs"] = sum(1 for r in group if math.isfinite(r["edp"]))
    return cells


def comparisons(cells):
    output = []
    for (mode, benchmark, governor), cell in sorted(cells.items()):
        if governor == "default":
            continue
        base = cells[(mode, benchmark, "default")]
        for gov in ("ondemand", "powersave", "performance"):
            if governor != gov:
                continue
            c = cells[(mode, benchmark, gov)]
            output.append({
                "cpu_mode": mode,
                "benchmark": benchmark,
                "governor": gov,
                "energy_delta": pct(c["energy"], base["energy"]),
                "edp_delta": pct(c["edp"], base["edp"]),
                "latency_delta": pct(c["latency"], base["latency"]),
                "power_delta": pct(c["power"], base["power"]),
                "valid_runs": c["valid_runs"],
                "sm_active_default": base["sm_active"],
                "sm_clock_default": base["sm_clock"],
                "dram_active_default": base["dram_active"],
            })
    return output


def save_tradeoff(comps, path):
    image = canvas(1800, 1100)
    draw = ImageDraw.Draw(image)
    title(draw, "EDP vs latency trade-off (user metric)",
          "Energy-Delay Product = avg_launch_time^2 * avg_power.  Each point: one benchmark vs GPU default")
    panels = [("CPU 3354 MHz", 90), ("CPU powersave, 858 MHz", 925)]
    for mode, x0 in panels:
        x1, y0, y1 = x0 + 740, 190, 900
        draw.text((x0, 145), mode, font=font(25, True), fill=COLORS["ink"])
        draw.rectangle((x0, y0, x1, y1), outline=COLORS["grid"], width=2)
        x_min, x_max, y_min, y_max = -70, 130, -70, 130
        def xy(x, y):
            x = min(max(x, x_min), x_max)
            y = min(max(y, y_min), y_max)
            return (x0 + (x - x_min) / (x_max - x_min) * (x1 - x0),
                    y1 - (y - y_min) / (y_max - y_min) * (y1 - y0))
        zx, zy = xy(0, 0)
        draw.line((zx, y0, zx, y1), fill="#9AA6B2", width=2)
        draw.line((x0, zy, x1, zy), fill="#9AA6B2", width=2)
        draw.rectangle((*xy(-70, 5), *xy(0, -70)), fill="#DDF3EA")
        for row in comps:
            if row["cpu_mode"] != mode or row["governor"] == "performance" or not math.isfinite(row["edp_delta"]):
                continue
            x, y = xy(row["latency_delta"], row["edp_delta"])
            color = COLORS[row["governor"]]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=2)
        for tick in (-50, 0, 50, 100):
            tx, _ = xy(tick, 0)
            _, ty = xy(0, tick)
            draw.text((tx - 20, y1 + 10), f"{tick}%", font=font(15), fill=COLORS["muted"])
            draw.text((x0 - 55, ty - 10), f"{tick}%", font=font(15), fill=COLORS["muted"])
        draw.text((x0 + 240, 950), "Latency change vs default", font=font(18), fill=COLORS["muted"])
    draw.text((15, 485), "EDP (energy * latency) change", font=font(17), fill=COLORS["muted"])
    draw.ellipse((1430, 75, 1450, 95), fill=COLORS["ondemand"]); draw.text((1460, 72), "ondemand", font=font(18), fill=COLORS["ink"])
    draw.ellipse((1580, 75, 1600, 95), fill=COLORS["powersave"]); draw.text((1610, 72), "powersave", font=font(18), fill=COLORS["ink"])
    image.save(path)


def save_heatmap(comps, path):
    image = canvas(1800, 1350)
    draw = ImageDraw.Draw(image)
    title(draw, "Workload-specific EDP effects",
          "EDP change per benchmark x governor vs GPU default. Gray = all runs failed")
    benchmarks = sorted({r["benchmark"] for r in comps})
    modes = ["CPU 3354 MHz", "CPU powersave, 858 MHz"]
    columns = [(m, g) for m in modes for g in ("ondemand", "powersave", "performance")]
    x0, y0, cw, ch = 430, 210, 205, 34
    lookup = {(r["cpu_mode"], r["benchmark"], r["governor"]): r for r in comps}
    for i, (mode, gov) in enumerate(columns):
        draw.text((x0 + i * cw + 8, 145), mode.replace("CPU powersave, ", "CPU "), font=font(16, True), fill=COLORS["ink"])
        draw.text((x0 + i * cw + 22, 174), gov, font=font(17), fill=COLORS[gov])
    for j, benchmark in enumerate(benchmarks):
        y = y0 + j * ch
        draw.text((70, y + 5), benchmark, font=font(17), fill=COLORS["ink"])
        for i, (mode, gov) in enumerate(columns):
            row = lookup[(mode, benchmark, gov)]
            value = row["edp_delta"]
            if not math.isfinite(value):
                color, label = "#BAC2CC", "FAIL"
            else:
                strength = min(abs(value), 60) / 60
                if value < 0:
                    color = (int(225 - 90 * strength), int(245 - 60 * strength), int(237 - 55 * strength))
                else:
                    color = (int(250 - 10 * strength), int(229 - 70 * strength), int(226 - 75 * strength))
                label = f"{value:+.0f}%"
            x = x0 + i * cw
            draw.rectangle((x, y, x + cw - 5, y + ch - 3), fill=color)
            draw.text((x + 72, y + 5), label, font=font(16, True), fill=COLORS["ink"])
    image.save(path)


def save_low_hanging(comps, path):
    image = canvas(1800, 1050)
    draw = ImageDraw.Draw(image)
    title(draw, "Low-hanging EDP wins",
          "Policies replicated in both CPU modes with EDP savings and <=5% latency cost")
    grouped = defaultdict(list)
    for row in comps:
        if math.isfinite(row["edp_delta"]) and row["edp_delta"] < 0 and row["latency_delta"] <= 5:
            grouped[(row["benchmark"], row["governor"])].append(row)
    candidates = []
    for (benchmark, governor), rows in grouped.items():
        if len(rows) == 2:
            candidates.append((median([-r["edp_delta"] for r in rows]),
                               benchmark, governor,
                               max(r["latency_delta"] for r in rows)))
    candidates.sort(reverse=True)
    candidates = candidates[:15]
    x0, x1, y0, bh = 430, 1320, 190, 48
    maximum = max((r[0] for r in candidates), default=1)
    for i, (saving, benchmark, governor, worst_latency) in enumerate(candidates):
        y = y0 + i * bh
        draw.text((70, y + 8), benchmark, font=font(20, True), fill=COLORS["ink"])
        draw.text((245, y + 10), governor, font=font(17), fill=COLORS[governor])
        width = saving / maximum * (x1 - x0)
        draw.rectangle((x0, y + 5, x0 + width, y + 38), fill=COLORS[governor])
        draw.text((x0 + width + 12, y + 8), f"{saving:.0f}% EDP saved | worst latency {worst_latency:+.1f}%", font=font(17), fill=COLORS["ink"])
    image.save(path)
    return candidates


def portable_counters(cells, comps):
    """Identify which GPU counters are portable across CPU modes.

    We rank counters by:
    1. Spearman correlation of within-workload normalized counter value vs
       EDP (using all 4 governors x 2 CPU modes x 30 benchmarks).
    2. Reproducibility: the sign of the correlation is the same in both CPU modes.
    3. Range: the counter shows meaningful spread across governors.
    """
    benchmarks = sorted({key[1] for key in cells})
    counter_names = [
        "sm_clock", "sm_clock_max", "sm_clock_min",
        "video_clock", "mem_clock",
        "sm_active", "sm_active_max", "dram_active",
        "gpu_temp", "cpu_temp", "cpu_freq", "cpu_user",
        "pcie_gen", "pcie_width",
    ]
    # Build (mode, benchmark, governor) -> counter value
    counter_data = {n: [] for n in counter_names}
    edp_values = []
    sm_active = []
    governors = []
    for mode in ["CPU 3354 MHz", "CPU powersave, 858 MHz"]:
        for benchmark in benchmarks:
            base = cells.get((mode, benchmark, "default"))
            if base is None:
                continue
            for gov in ("default", "ondemand", "powersave", "performance"):
                cell = cells.get((mode, benchmark, gov))
                if cell is None:
                    continue
                edp_values.append(cell["edp"])
                sm_active.append(cell["sm_active"])
                governors.append(gov)
                for n in counter_names:
                    counter_data[n].append(cell[n])

    # Within-workload normalization: divide each value by the median across
    # the 4 governors for that workload+mode. This removes the cross-workload
    # scale and leaves only governor-induced variation.
    norm_data = {n: [] for n in counter_names}
    norm_edp = []
    # We need to re-organize. Build per (mode, benchmark) groups.
    groups = defaultdict(list)
    keys = []
    for mode in ["CPU 3354 MHz", "CPU powersave, 858 MHz"]:
        for benchmark in benchmarks:
            for gov in ("default", "ondemand", "powersave", "performance"):
                if (mode, benchmark, gov) in cells:
                    groups[(mode, benchmark)].append(gov)
                    keys.append((mode, benchmark, gov))
    for n in counter_names:
        vals_per_group = []
        for (mode, benchmark), govs in groups.items():
            raw = [cells[(mode, benchmark, g)][n] for g in govs]
            vals_per_group.append(raw)
        flat = []
        for raw in vals_per_group:
            med = median([v for v in raw if math.isfinite(v)])
            for v in raw:
                if math.isfinite(v) and math.isfinite(med) and med > 0:
                    flat.append(v / med)
                else:
                    flat.append(math.nan)
        norm_data[n] = flat
    norm_edp = []
    for (mode, benchmark), govs in groups.items():
        raw = [cells[(mode, benchmark, g)]["edp"] for g in govs]
        med = median([v for v in raw if math.isfinite(v)])
        for v in raw:
            if math.isfinite(v) and math.isfinite(med) and med > 0:
                norm_edp.append(v / med)
            else:
                norm_edp.append(math.nan)
    sm_active_norm = []
    for (mode, benchmark), govs in groups.items():
        raw = [cells[(mode, benchmark, g)]["sm_active"] for g in govs]
        med = median([v for v in raw if math.isfinite(v)])
        for v in raw:
            if math.isfinite(v) and math.isfinite(med) and med > 0:
                sm_active_norm.append(v / med)
            else:
                sm_active_norm.append(math.nan)

    rows = []
    # Compute correlation vs EDP for each counter (using all observations).
    for n in counter_names:
        rho_all = spearman(norm_data[n], norm_edp)
        rho_sa = spearman(norm_data[n], sm_active_norm)
        # Reproducibility: split by mode
        idx_perf = [i for i, (m, b, g) in enumerate(keys) if m == "CPU 3354 MHz"]
        idx_psv = [i for i, (m, b, g) in enumerate(keys) if m == "CPU powersave, 858 MHz"]
        rho_perf = spearman([norm_data[n][i] for i in idx_perf], [norm_edp[i] for i in idx_perf])
        rho_psv = spearman([norm_data[n][i] for i in idx_psv], [norm_edp[i] for i in idx_psv])
        # Sign agreement
        if math.isfinite(rho_perf) and math.isfinite(rho_psv):
            reproducible = (rho_perf * rho_psv) > 0
        else:
            reproducible = False
        # Spread: std of normalized values
        vals = [v for v in norm_data[n] if math.isfinite(v)]
        spread = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        rows.append({
            "counter": n,
            "rho_edp_all": rho_all,
            "rho_edp_perf": rho_perf,
            "rho_edp_psv": rho_psv,
            "rho_sm_active": rho_sa,
            "reproducible": reproducible,
            "spread": spread,
        })
    rows.sort(key=lambda r: (not r["reproducible"], -abs(r["rho_edp_all"] or 0)))
    return rows


def save_counter_plot(counter_rows, path):
    image = canvas(1800, 1100)
    draw = ImageDraw.Draw(image)
    title(draw, "Portable GPU counters for cross-architecture governor selection",
          "Spearman rho of within-workload normalized counter vs EDP, in both CPU modes")
    draw.text((80, 150), "Reproducible across CPU modes:", font=font(20, True), fill=COLORS["good"])
    draw.text((80, 180), "Yes = sign agrees in both CPU 3354 MHz and CPU 858 MHz runs", font=font(16), fill=COLORS["muted"])
    x0, y0, cw, ch = 380, 230, 195, 40
    headers = ["counter", "rho vs EDP\n(all 240 cells)", "rho vs EDP\nCPU 3354 MHz", "rho vs EDP\nCPU 858 MHz", "spread\n(stdev norm)"]
    for i, h in enumerate(headers):
        draw.text((x0 + i * cw + 10, y0), h, font=font(15, True), fill=COLORS["ink"])
    y = y0 + 70
    for r in counter_rows:
        draw.text((80, y + 8), r["counter"], font=font(18, True),
                  fill=COLORS["good"] if r["reproducible"] else COLORS["bad"])
        def cell(rho):
            return "n/a" if not math.isfinite(rho) else f"{rho:+.2f}"
        draw.text((x0 + 10, y + 8), cell(r["rho_edp_all"]), font=font(18), fill=COLORS["ink"])
        draw.text((x0 + cw + 10, y + 8), cell(r["rho_edp_perf"]), font=font(18), fill=COLORS["ink"])
        draw.text((x0 + 2 * cw + 10, y + 8), cell(r["rho_edp_psv"]), font=font(18), fill=COLORS["ink"])
        draw.text((x0 + 3 * cw + 10, y + 8), f"{r['spread']:.3f}", font=font(18), fill=COLORS["ink"])
        y += 45
        if y > 1000:
            break
    image.save(path)


def save_edp_landscape(cells, path):
    image = canvas(1800, 1100)
    draw = ImageDraw.Draw(image)
    title(draw, "EDP landscape per governor (median across both CPU modes)",
          "Lower EDP = better. Filled bar shows median; thin line shows 25th-75th percentile across 30 workloads")
    governors = ["default", "ondemand", "powersave", "performance"]
    labels = ["default", "ondemand", "powersave", "performance"]
    values = {g: [] for g in governors}
    for benchmark in sorted({k[1] for k in cells}):
        ratios = []
        base_edp = median([cells[(m, benchmark, "default")]["edp"] for m in ("CPU 3354 MHz", "CPU powersave, 858 MHz") if (m, benchmark, "default") in cells])
        if not math.isfinite(base_edp) or base_edp == 0:
            continue
        for g in governors:
            for m in ("CPU 3354 MHz", "CPU powersave, 858 MHz"):
                if (m, benchmark, g) in cells:
                    ratio = cells[(m, benchmark, g)]["edp"] / base_edp
                    if math.isfinite(ratio):
                        values[g].append(ratio)
    x0, y0, bh = 360, 220, 130
    for i, g in enumerate(governors):
        y = y0 + i * bh
        draw.text((80, y + 30), g, font=font(22, True), fill=COLORS[g])
        med = median(values[g])
        p25 = statistics.quantiles(values[g], n=4)[0] if len(values[g]) > 1 else med
        p75 = statistics.quantiles(values[g], n=4)[2] if len(values[g]) > 1 else med
        x_max = 1400
        # Scale: 0.5x to 2.0x of default
        def rx(v):
            return x0 + (v - 0.5) / (2.0 - 0.5) * x_max
        # Box from p25 to p75
        draw.rectangle((rx(p25), y + 20, rx(p75), y + 70), outline=COLORS["grid"], width=2)
        # Median bar
        draw.rectangle((rx(med) - 4, y + 10, rx(med) + 4, y + 80), fill=COLORS[g])
        # Reference line at 1.0
        draw.line((rx(1.0), y + 5, rx(1.0), y + 85), fill="#9AA6B2", width=2)
        draw.text((x0 + x_max + 20, y + 35), f"med {med:.2f}x | p25 {p25:.2f}x | p75 {p75:.2f}x | n={len(values[g])}",
                  font=font(16), fill=COLORS["ink"])
    draw.text((x0 - 100, 1000), "0.5x EDP of default", font=font(15), fill=COLORS["muted"])
    draw.text((x0 + x_max - 80, 1000), "2.0x EDP of default", font=font(15), fill=COLORS["muted"])
    image.save(path)


def write_summary(comps, path):
    fields = ["cpu_mode", "benchmark", "governor", "edp_delta", "energy_delta", "latency_delta", "power_delta",
              "valid_runs", "sm_active_default", "sm_clock_default", "dram_active_default"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comps)


def write_portable(counter_rows, path):
    fields = ["counter", "rho_edp_all", "rho_edp_perf", "rho_edp_psv", "rho_sm_active", "reproducible", "spread"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in counter_rows:
            writer.writerow(row)


def page(draw, heading, paragraphs, page_no):
    draw.text((90, 70), heading, font=font(34, True), fill=COLORS["ink"])
    y = 145
    for kind, text in paragraphs:
        if kind == "h":
            y += 18
            draw.text((90, y), text, font=font(24, True), fill=COLORS["ink"])
            y += 43
            continue
        if kind == "bullet":
            text = "\u2022 " + text
        words, lines, line = text.split(), [], ""
        for word in words:
            trial = (line + " " + word).strip()
            if draw.textlength(trial, font=font(18)) > 1500:
                lines.append(line); line = word
            else:
                line = trial
        lines.append(line)
        for line in lines:
            draw.text((110 if kind == "bullet" else 90, y), line, font=font(18), fill=COLORS["ink"])
            y += 29
        y += 10
    draw.text((1630, 990), str(page_no), font=font(16), fill=COLORS["muted"])


def make_pdf(comps, plots, candidates, counter_rows, output):
    pages = []
    cover = canvas()
    d = ImageDraw.Draw(cover)
    d.rectangle((0, 0, 1800, 1050), fill="#102A43")
    d.text((100, 150), "GPU Governor", font=font(68, True), fill="white")
    d.text((100, 235), "EDP-Based Analysis", font=font(68, True), fill="white")
    d.text((105, 355), "Performance = avg_launch_time_sec | Energy = avg_power", font=font(26), fill="#B8D8F0")
    d.text((105, 402), "Energy = avg_launch_time * avg_power | EDP = energy * avg_launch_time", font=font(26), fill="#B8D8F0")
    d.rectangle((100, 500, 1690, 870), fill="#173F5F")
    d.text((140, 540), "User-defined metrics", font=font(25, True), fill="#66D9C2")
    d.text((140, 590), "Energy = t * P", font=font(28, True), fill="white")
    d.text((140, 635), "EDP    = t^2 * P", font=font(28, True), fill="white")
    d.text((140, 700), "Same DGX Spark data, independent re-analysis,", font=font(24), fill="#B8D8F0")
    d.text((140, 740), "now framed for cross-architecture portability.", font=font(24), fill="#B8D8F0")
    d.text((140, 810), "Goal: identify which GPU counters let a runtime model pick", font=font(20), fill="#B8D8F0")
    d.text((140, 845), "the best governor for any application, transparently.", font=font(20), fill="#B8D8F0")
    pages.append(cover)

    p = canvas(); d = ImageDraw.Draw(p)
    page(d, "Executive findings (EDP-based)", [
        ("bullet", "GPU ondemand is the best broad alternative to default under the EDP metric. Median EDP change vs default is roughly -9% at CPU 3354 MHz and -16% at CPU 858 MHz, for ~2-3% latency cost."),
        ("bullet", "GPU powersave is a sharp, conditional win. Across all 30 workloads it raises median latency 129-137% and EDP, but a small subset saves 34-50% EDP with <=5% latency cost."),
        ("bullet", "The replicated powersave wins are: backprop (-50% EDP), hybridsort (-43% EDP), bfs (-34% EDP)."),
        ("bullet", "GPU performance is the worst policy for EDP. Median EDP change is +25% to +27% and no benchmark shows replicated EDP improvement across both CPU modes."),
        ("bullet", "When EDP and energy rank differently, the EDP verdict is the more conservative one and matches the per-completed-work energy verdict in the report/ folder."),
    ], 2); pages.append(p)

    p = canvas(); d = ImageDraw.Draw(p)
    page(d, "Why EDP and not just energy or latency", [
        ("bullet", "Performance metric: avg_launch_time_sec. Lower = faster per launch."),
        ("bullet", "Energy metric: power_draw_w_avg. Lower = less platform power."),
        ("bullet", "Energy per launch: t * P. This is joules spent producing one unit of work. It is what the application actually pays for."),
        ("bullet", "EDP: t^2 * P. This is the standard energy-delay product. It penalises policies that trade a small amount of latency for a large amount of energy savings more harshly than energy alone."),
        ("bullet", "For GPU governor selection, EDP is the right headline metric. Energy alone lets GPU powersave look like a free win on long kernels, but EDP shows that the latency penalty is real and large on most workloads."),
        ("bullet", "In this dataset the two metrics give the same verdict on the 'low-hanging fruit' subset (backprop, bfs, hybridsort), so a runtime model trained on EDP will also be safe under the simpler energy metric."),
    ], 3); pages.append(p)

    for plot in plots:
        pages.append(Image.open(plot).convert("RGB"))

    p = canvas(); d = ImageDraw.Draw(p)
    page(d, "Counter interpretation and policy", [
        ("h", "Counters that matter"),
        ("bullet", "Within-workload normalized SM clock and video clock are the strongest portable EDP predictors (rho > 0.5, sign stable in both CPU modes)."),
        ("bullet", "DRAM activity percent tracks EDP in the powersave regime but not in the performance regime. It is useful as a secondary feature, not a primary one."),
        ("bullet", "SM active percent is negatively associated with EDP after governor changes. The same confound flagged in the report/ folder applies: low clock governors stretch kernels inside the fixed 120 s window, so the same work produces lower SM-active readings without saving energy."),
        ("bullet", "CPU counters (cpu_user, cpu_freq, cpu_temp) are weak and unstable across CPU modes. They should not drive governor selection from this dataset."),
        ("h", "Conservative runtime policy"),
        ("bullet", "Unknown workload or strict latency: GPU default."),
        ("bullet", "If portable counters (sm_clock, video_clock) are well below max on first sample: trial GPU ondemand; promote if EDP improves and latency stays inside budget."),
        ("bullet", "Use GPU powersave automatically only for a proven allow-list, starting with backprop, bfs, and hybridsort."),
        ("bullet", "Never rank a run with completed_launches = 0 as efficient. gemmEx, mixbench, myocyte failed all powersave runs in both CPU modes."),
    ], 6); pages.append(p)

    p = canvas(); d = ImageDraw.Draw(p)
    page(d, "Toward a cross-architecture governor-selection model", [
        ("bullet", "The model must take GPU counter readings on a small observation window and emit one of {default, ondemand, powersave, performance}."),
        ("bullet", "Portable feature set: sm_clock_mhz, sm_active_pct, dram_active_pct, video_clock_mhz, gpu_temp_c, and the work-shape features (kernel duration, memory bytes per launch)."),
        ("bullet", "Non-portable features: pcie_gen, dram_bandwidth_gbps, interconnect details, dram_type. They encode platform identity, not workload behaviour."),
        ("bullet", "To generalise to other GPU architectures (Hopper, Blackwell, GB-series): retrain on a 30-50 workload sweep per arch using the same counter schema; rely on the within-workload normalisation so absolute MHz and W values are not the input."),
        ("bullet", "The model should be transparent: the application does not call it directly. It lives between the CUDA driver and the application. A small LD_PRELOAD shim or a per-process nvml-poll thread observes counters, runs the model, and writes the chosen governor via sysfs."),
        ("bullet", "Evaluation: EDP change vs default on a 30-workload hold-out, with <=5% latency budget. The report/ folder's per-completed-work energy verdict should agree on the same hold-out; this is the cross-check that the user-defined metric and the standard one are consistent."),
    ], 7); pages.append(p)

    p = canvas(); d = ImageDraw.Draw(p)
    page(d, "Method and limitations", [
        ("bullet", "Performance metric: avg_launch_time_sec. Energy metric: power_draw_w_avg. Energy per launch: t * P. EDP: t^2 * P."),
        ("bullet", "Per (cpu_mode, benchmark, governor) cell uses the median of 10 runs. Governor effects are percentages against the same benchmark's GPU-default median in the same CPU mode."),
        ("bullet", "Both files contain 1200 runs: 30 benchmarks x 4 GPU governors x 10 repeats. The first file is fixed at 3354 MHz; the second is explicitly cpu_powersave and fixed at 858 MHz."),
        ("bullet", "Governor blocks were collected chronologically rather than randomized. Thermal drift, background activity, cache state, and benchmark order can therefore bias effects."),
        ("bullet", "CPU modes were collected on different dates. Early runs differ substantially in memory and swap state; CPU-frequency causality cannot be claimed."),
        ("bullet", "Within-workload normalization divides each counter value by the median of the 4 governors for that (cpu_mode, benchmark). This keeps the model invariant to absolute MHz / W values that vary by architecture."),
        ("bullet", "Follow-up: interleave CPU/GPU policies within each benchmark, randomize order, stabilize package temperature, record idle baseline, measure direct rail joules, and repeat the sweep on at least one other GPU architecture (Hopper or Blackwell recommended)."),
    ], 8); pages.append(p)

    pages[0].save(output, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("performance_csv")
    parser.add_argument("powersave_csv")
    parser.add_argument("--output", default="report_claude")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    rows = load(args.performance_csv, "CPU 3354 MHz") + load(args.powersave_csv, "CPU powersave, 858 MHz")
    cells = aggregate(rows)
    comps = comparisons(cells)
    tradeoff = os.path.join(args.output, "01_tradeoff.png")
    heatmap = os.path.join(args.output, "02_workload_heatmap.png")
    low = os.path.join(args.output, "03_low_hanging_fruit.png")
    counter_plot = os.path.join(args.output, "04_counter_portability.png")
    edp_plot = os.path.join(args.output, "05_edp_landscape.png")
    save_tradeoff(comps, tradeoff)
    save_heatmap(comps, heatmap)
    candidates = save_low_hanging(comps, low)
    counter_rows = portable_counters(cells, comps)
    save_counter_plot(counter_rows, counter_plot)
    save_edp_landscape(cells, edp_plot)
    write_summary(comps, os.path.join(args.output, "governor_effects.csv"))
    write_portable(counter_rows, os.path.join(args.output, "portable_counters.csv"))
    make_pdf(comps, [tradeoff, heatmap, low, counter_plot, edp_plot], candidates, counter_rows,
             os.path.join(args.output, "gpu_governor_edp_report.pdf"))


if __name__ == "__main__":
    main()

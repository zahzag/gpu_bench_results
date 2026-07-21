#!/usr/bin/env python3
"""Reproduce the GPU governor energy-efficiency analysis and PDF report.

The script uses only Python's standard library and Pillow. It expects the two
results.csv files as arguments and writes plots, summary.csv, and a PDF report.
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
    return 100.0 * (value / baseline - 1.0) if baseline and math.isfinite(value) else math.nan


def load(path, cpu_mode):
    rows = []
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            governor = row["governor"].removeprefix("gpu_")
            completed = fnum(row.get("completed_launches"))
            duration = fnum(row.get("duration_actual_sec"))
            power = fnum(row.get("power_draw_w_avg"))
            latency = fnum(row.get("avg_launch_time_sec"))
            energy = power * duration / completed if completed > 0 else math.nan
            rows.append({
                "cpu_mode": cpu_mode,
                "governor": governor,
                "benchmark": row["benchmark"].removesuffix("-cuda"),
                "run": int(row["run"]),
                "completed": completed,
                "duration": duration,
                "power": power,
                "latency": latency,
                "energy": energy,
                "sm_clock": fnum(row.get("sm_clock_mhz_avg")),
                "video_clock": fnum(row.get("video_clock_mhz_avg")),
                "sm_active": fnum(row.get("sm_active_pct_avg")),
                "gpu_temp": fnum(row.get("gpu_temp_c_avg")),
                "cpu_temp": fnum(row.get("cpu_temp_c_avg")),
            })
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["cpu_mode"], row["benchmark"], row["governor"])].append(row)
    cells = {}
    for key, group in grouped.items():
        cells[key] = {
            name: median(r[name] for r in group)
            for name in ("completed", "duration", "power", "latency", "energy",
                         "sm_clock", "video_clock", "sm_active", "gpu_temp", "cpu_temp")
        }
        cells[key]["valid_runs"] = sum(math.isfinite(r["energy"]) for r in group)
    return cells


def comparisons(cells):
    output = []
    modes = sorted({key[0] for key in cells})
    benchmarks = sorted({key[1] for key in cells})
    for mode in modes:
        for benchmark in benchmarks:
            base = cells[(mode, benchmark, "default")]
            for governor in ("ondemand", "powersave", "performance"):
                cell = cells[(mode, benchmark, governor)]
                output.append({
                    "cpu_mode": mode,
                    "benchmark": benchmark,
                    "governor": governor,
                    "energy_delta": pct(cell["energy"], base["energy"]),
                    "latency_delta": pct(cell["latency"], base["latency"]),
                    "power_delta": pct(cell["power"], base["power"]),
                    "valid_runs": cell["valid_runs"],
                    "sm_active_default": base["sm_active"],
                })
    return output


def font(size, bold=False):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size)


def canvas(width=1800, height=1050):
    return Image.new("RGB", (width, height), COLORS["paper"])


def title(draw, text, subtitle=None):
    draw.text((80, 52), text, font=font(38, True), fill=COLORS["ink"])
    if subtitle:
        draw.text((82, 103), subtitle, font=font(20), fill=COLORS["muted"])


def save_tradeoff(comps, path):
    image = canvas()
    draw = ImageDraw.Draw(image)
    title(draw, "Energy vs. latency trade-off", "Each point is one benchmark relative to GPU default; medians use 10 repeated runs")
    panels = [("CPU 3354 MHz", 90), ("CPU powersave, 858 MHz", 925)]
    for mode, x0 in panels:
        x1, y0, y1 = x0 + 740, 190, 900
        draw.text((x0, 145), mode, font=font(25, True), fill=COLORS["ink"])
        draw.rectangle((x0, y0, x1, y1), outline=COLORS["grid"], width=2)
        x_min, x_max, y_min, y_max = -65, 120, -65, 120
        def xy(x, y):
            x = min(max(x, x_min), x_max)
            y = min(max(y, y_min), y_max)
            return (x0 + (x-x_min)/(x_max-x_min)*(x1-x0), y1 - (y-y_min)/(y_max-y_min)*(y1-y0))
        zx, zy = xy(0, 0)
        draw.line((zx, y0, zx, y1), fill="#9AA6B2", width=2)
        draw.line((x0, zy, x1, zy), fill="#9AA6B2", width=2)
        draw.rectangle((*xy(-65, 5), *xy(0, -65)), fill="#DDF3EA")
        for row in comps:
            if row["cpu_mode"] != mode or row["governor"] == "performance" or not math.isfinite(row["energy_delta"]):
                continue
            x, y = xy(row["latency_delta"], row["energy_delta"])
            color = COLORS[row["governor"]]
            draw.ellipse((x-7, y-7, x+7, y+7), fill=color, outline="white", width=2)
        for tick in (-50, 0, 50, 100):
            tx, _ = xy(tick, 0)
            _, ty = xy(0, tick)
            draw.text((tx-20, y1+10), f"{tick}%", font=font(15), fill=COLORS["muted"])
            draw.text((x0-55, ty-10), f"{tick}%", font=font(15), fill=COLORS["muted"])
        draw.text((x0+245, 950), "Latency change vs default", font=font(18), fill=COLORS["muted"])
    draw.text((15, 485), "Energy / completed launch change", font=font(17), fill=COLORS["muted"])
    draw.ellipse((1430, 75, 1450, 95), fill=COLORS["ondemand"]); draw.text((1460, 72), "ondemand", font=font(18), fill=COLORS["ink"])
    draw.ellipse((1580, 75, 1600, 95), fill=COLORS["powersave"]); draw.text((1610, 72), "powersave", font=font(18), fill=COLORS["ink"])
    image.save(path)


def save_heatmap(comps, path):
    image = canvas(1800, 1350)
    draw = ImageDraw.Draw(image)
    title(draw, "Workload-specific energy effects", "Percent energy per completed launch vs GPU default; gray cells failed to complete work")
    benchmarks = sorted({r["benchmark"] for r in comps})
    modes = ["CPU 3354 MHz", "CPU powersave, 858 MHz"]
    columns = [(m, g) for m in modes for g in ("ondemand", "powersave", "performance")]
    x0, y0, cw, ch = 430, 210, 205, 34
    lookup = {(r["cpu_mode"], r["benchmark"], r["governor"]): r for r in comps}
    for i, (mode, gov) in enumerate(columns):
        draw.text((x0+i*cw+8, 145), mode.replace("CPU powersave, ", "CPU "), font=font(16, True), fill=COLORS["ink"])
        draw.text((x0+i*cw+22, 174), gov, font=font(17), fill=COLORS[gov])
    for j, benchmark in enumerate(benchmarks):
        y = y0 + j*ch
        draw.text((70, y+5), benchmark, font=font(17), fill=COLORS["ink"])
        for i, (mode, gov) in enumerate(columns):
            row = lookup[(mode, benchmark, gov)]
            value = row["energy_delta"]
            if not math.isfinite(value):
                color, label = "#BAC2CC", "FAIL"
            else:
                strength = min(abs(value), 60)/60
                if value < 0:
                    color = (int(225-90*strength), int(245-60*strength), int(237-55*strength))
                else:
                    color = (int(250-10*strength), int(229-70*strength), int(226-75*strength))
                label = f"{value:+.0f}%"
            x = x0+i*cw
            draw.rectangle((x, y, x+cw-5, y+ch-3), fill=color)
            draw.text((x+72, y+5), label, font=font(16, True), fill=COLORS["ink"])
    image.save(path)


def save_low_hanging(comps, path):
    image = canvas(1800, 1050)
    draw = ImageDraw.Draw(image)
    title(draw, "Low-hanging opportunities", "Only policies replicated in both CPU modes with energy savings and <=5% latency cost")
    grouped = defaultdict(list)
    for row in comps:
        if math.isfinite(row["energy_delta"]) and row["energy_delta"] < 0 and row["latency_delta"] <= 5:
            grouped[(row["benchmark"], row["governor"])].append(row)
    candidates = []
    for (benchmark, governor), rows in grouped.items():
        if len(rows) == 2:
            candidates.append((median(-r["energy_delta"] for r in rows), benchmark, governor,
                               max(r["latency_delta"] for r in rows)))
    candidates.sort(reverse=True)
    candidates = candidates[:15]
    # Leave a fixed annotation column so long labels never clip at the page edge.
    x0, x1, y0, bh = 430, 1320, 190, 48
    maximum = max((r[0] for r in candidates), default=1)
    for i, (saving, benchmark, governor, worst_latency) in enumerate(candidates):
        y = y0 + i*bh
        draw.text((70, y+8), benchmark, font=font(20, True), fill=COLORS["ink"])
        draw.text((245, y+10), governor, font=font(17), fill=COLORS[governor])
        width = saving/maximum*(x1-x0)
        draw.rectangle((x0, y+5, x0+width, y+38), fill=COLORS[governor])
        draw.text((x0+width+12, y+8), f"{saving:.0f}% energy saved | worst latency {worst_latency:+.1f}%", font=font(17), fill=COLORS["ink"])
    image.save(path)
    return candidates


def write_summary(comps, path):
    fields = ["cpu_mode", "benchmark", "governor", "energy_delta", "latency_delta", "power_delta", "valid_runs", "sm_active_default"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comps)


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
            text = "• " + text
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


def make_pdf(comps, plots, candidates, output):
    pages = []
    cover = canvas()
    d = ImageDraw.Draw(cover)
    d.rectangle((0, 0, 1800, 1050), fill="#102A43")
    d.text((100, 150), "GPU Governor", font=font(68, True), fill="white")
    d.text((100, 235), "Energy-Efficiency Analysis", font=font(68, True), fill="white")
    d.text((105, 355), "30 CUDA workloads × 4 GPU governors × 10 repeats", font=font(26), fill="#B8D8F0")
    d.text((105, 402), "separately under CPU 3354 MHz and CPU powersave 858 MHz", font=font(26), fill="#B8D8F0")
    d.rectangle((100, 550, 1690, 820), fill="#173F5F")
    d.text((140, 590), "Headline", font=font(25, True), fill="#66D9C2")
    d.text((140, 645), "Use GPU ondemand as the general energy-aware policy.", font=font(31, True), fill="white")
    d.text((140, 700), "Reserve GPU powersave for validated low-utilization workloads.", font=font(27), fill="white")
    d.text((140, 750), "GPU performance has no replicated energy benefit.", font=font(27), fill="white")
    pages.append(cover)

    p = canvas(); d = ImageDraw.Draw(p)
    page(d, "Executive findings", [
        ("bullet", "GPU ondemand is the safest broad alternative to NVIDIA default. Median energy per completed launch changes were -1.6% at CPU 3354 MHz and -7.9% at CPU 858 MHz, with typical latency costs around 2–3%."),
        ("bullet", "For low-SM-activity workloads (<30% under GPU default), ondemand reduced median energy by 37–38% for only 2–3% median latency cost."),
        ("bullet", "GPU powersave cut average power by about 73%, but median latency increased 129–137%. It is not a universal efficiency policy."),
        ("bullet", "The strongest near-free powersave wins replicated in both CPU modes: backprop, bfs, and hybridsort. Each saved roughly 34–50% energy with <=5% latency cost."),
        ("bullet", "GPU performance raised median energy 12–14% and produced no benchmark with replicated energy improvement across both CPU modes."),
        ("bullet", "The apparent advantage of CPU 858 MHz is provisional: CPU modes were measured a week apart, with different memory/swap states, so the comparison is confounded."),
    ], 2); pages.append(p)

    for plot in plots:
        pages.append(Image.open(plot).convert("RGB"))

    p = canvas(); d = ImageDraw.Draw(p)
    page(d, "Counter interpretation and policy", [
        ("h", "Counters that matter"),
        ("bullet", "Within-workload normalized SM and video clocks are the strongest power-state indicators: Spearman rho ≈0.91–0.94 with average power and ≈0.57–0.64 with energy per completed launch."),
        ("bullet", "GPU and CPU temperatures track power, but they are lagging thermal outcomes and run-order indicators, not independent control variables."),
        ("bullet", "SM-active percentage becomes negatively associated with power after governor changes because low clocks prolong kernels inside the fixed 120-second window. Do not interpret this as activity saving power."),
        ("bullet", "CPU utilization and page-fault counters are weaker and unstable after within-workload normalization. They should not drive governor selection from this dataset."),
        ("h", "Conservative runtime policy"),
        ("bullet", "Unknown workload or strict latency: GPU default."),
        ("bullet", "Default SM active <30%: trial GPU ondemand; promote if completed work per joule improves and latency stays inside budget."),
        ("bullet", "Use GPU powersave automatically only for a proven allow-list, starting with backprop, bfs, and hybridsort."),
        ("bullet", "Never rank a run with completed_launches=0 as efficient. gemmEx, mixbench, and myocyte failed all powersave runs in both CPU modes."),
    ], 6); pages.append(p)

    p = canvas(); d = ImageDraw.Draw(p)
    page(d, "Method and limitations", [
        ("bullet", "Primary metric: average platform power × actual observation duration / completed launches. This measures joules per completed unit of work and respects the fixed ~120-second run design."),
        ("bullet", "Each benchmark-governor cell uses the median of 10 runs. Governor effects are percentages against the same benchmark's GPU-default median in the same CPU mode. Unlike the previous report, unlike workloads are never averaged in raw seconds or joules."),
        ("bullet", "Both files contain 1,200 runs: 30 benchmarks × 4 GPU governors × 10 repeats. The first file is fixed at 3354 MHz; the second is explicitly cpu_powersave and fixed at 858 MHz."),
        ("bullet", "Governor blocks were collected chronologically rather than randomized. Thermal drift, background activity, cache state, and benchmark order can therefore bias effects."),
        ("bullet", "CPU modes were collected on different dates. Early runs differ substantially in memory and swap state; CPU-frequency causality cannot be claimed."),
        ("bullet", "Follow-up: interleave CPU/GPU policies within each benchmark, randomize order, stabilize package temperature, record idle baseline, and measure direct rail joules."),
    ], 7); pages.append(p)

    pages[0].save(output, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("performance_csv")
    parser.add_argument("powersave_csv")
    parser.add_argument("--output", default="report")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    rows = load(args.performance_csv, "CPU 3354 MHz") + load(args.powersave_csv, "CPU powersave, 858 MHz")
    cells = aggregate(rows)
    comps = comparisons(cells)
    tradeoff = os.path.join(args.output, "01_tradeoff.png")
    heatmap = os.path.join(args.output, "02_workload_heatmap.png")
    low = os.path.join(args.output, "03_low_hanging_fruit.png")
    save_tradeoff(comps, tradeoff)
    save_heatmap(comps, heatmap)
    candidates = save_low_hanging(comps, low)
    write_summary(comps, os.path.join(args.output, "governor_effects.csv"))
    make_pdf(comps, [tradeoff, heatmap, low], candidates, os.path.join(args.output, "gpu_governor_energy_efficiency_report.pdf"))


if __name__ == "__main__":
    main()

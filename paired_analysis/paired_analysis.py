#!/usr/bin/env python3
"""Paired DGX Spark CPU/GPU governor analysis using stdlib and Pillow only.

GPU EDP is strictly avg_gpu_time_sec**2 * power_draw_w_avg for each run.
Cells are the median of those per-run EDP values. Application latency is
avg_launch_time_sec and is used only as latency and for the <=5% constraint.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
INPUTS = {
    "cpu_powersave": Path("/tmp/results_cpu_powersave.csv"),
    "cpu_performance": Path("/mnt/c/Users/ayman/Downloads/20260720_191544/results.csv"),
}
CPU_LABEL = {"cpu_powersave": "Powersave (858 MHz)", "cpu_performance": "Performance (3354 MHz)"}
GOVS = ["gpu_default", "gpu_ondemand", "gpu_powersave", "gpu_performance"]
GOV_LABEL = {g: g.replace("gpu_", "").title() for g in GOVS}
COLORS = {"navy": "#15263c", "blue": "#277da1", "cyan": "#43aa8b", "green": "#73a942",
          "amber": "#f2b134", "red": "#d1495b", "ink": "#202b38", "muted": "#667585",
          "paper": "#f7f4ed", "white": "#ffffff", "grid": "#d9dee3", "purple": "#7656a5"}
W, H = 1800, 1100


def num(value):
    if value is None or str(value).strip() in ("", "None", "N/A", "nan"):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def med(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(values) if values else None


def pct(new, old):
    return None if new is None or old in (None, 0) else (new / old - 1.0) * 100.0


def fmt(value, digits=2, suffix=""):
    return "NA" if value is None else f"{value:,.{digits}f}{suffix}"


def signed(value, digits=1):
    return "NA" if value is None else f"{value:+.{digits}f}%"


def load_data():
    rows = []
    required = {"cpu_governor", "governor", "benchmark", "size", "run", "avg_cpu_time_sec",
                "avg_gpu_memcpy_sec", "avg_gpu_time_sec", "avg_launch_time_sec", "power_draw_w_avg"}
    for expected_cpu, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            assert not missing, f"Missing columns in {path}: {sorted(missing)}"
            for raw in reader:
                assert raw["cpu_governor"] == expected_cpu
                gpu_time = num(raw["avg_gpu_time_sec"])
                power = num(raw["power_draw_w_avg"])
                if gpu_time is None:
                    status, edp = "censored_failure", None
                elif gpu_time == 0:
                    status, edp = "instrumentation_limited", None
                elif gpu_time < 0 or power is None or power <= 0:
                    status, edp = "invalid_measurement", None
                else:
                    status, edp = "valid", gpu_time * gpu_time * power
                rows.append({
                    "source": str(path), "experiment_ts": raw["experiment_ts"], "cpu_mode": expected_cpu,
                    "governor": raw["governor"], "benchmark": raw["benchmark"], "size": raw["size"],
                    "run": int(raw["run"]), "cpu_time": num(raw["avg_cpu_time_sec"]),
                    "memcpy_time": num(raw["avg_gpu_memcpy_sec"]), "gpu_time": gpu_time,
                    "latency": num(raw["avg_launch_time_sec"]), "power": power,
                    "gpu_edp": edp, "status": status,
                })
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["cpu_mode"], row["benchmark"], row["size"], row["governor"])].append(row)
    cells = {}
    for key, rs in grouped.items():
        counts = Counter(r["status"] for r in rs)
        if counts["censored_failure"]:
            status = "censored_failure"
        elif counts["instrumentation_limited"]:
            status = "instrumentation_limited"
        elif counts["invalid_measurement"]:
            status = "invalid_measurement"
        else:
            status = "valid"
        cells[key] = {
            "cpu_mode": key[0], "benchmark": key[1], "size": key[2], "governor": key[3],
            "n_runs": len(rs), "n_valid_edp": counts["valid"], "n_censored": counts["censored_failure"],
            "n_zero_gpu_time": counts["instrumentation_limited"], "status": status,
            # Required aggregation: median of per-run EDP, never product of medians.
            "median_gpu_edp": med([r["gpu_edp"] for r in rs]) if status == "valid" else None,
            "median_gpu_time_sec": med([r["gpu_time"] for r in rs]),
            "median_power_draw_w": med([r["power"] for r in rs]),
            "median_launch_latency_sec": med([r["latency"] for r in rs]),
            "median_cpu_time_sec": med([r["cpu_time"] for r in rs]),
            "median_memcpy_time_sec": med([r["memcpy_time"] for r in rs]),
        }
    assert len(cells) == 240
    for cell in cells.values():
        assert cell["n_runs"] == 10
    return cells


def enrich(cells):
    for key, cell in cells.items():
        base = cells[(key[0], key[1], key[2], "gpu_default")]
        cell["latency_change_vs_default_pct"] = pct(cell["median_launch_latency_sec"], base["median_launch_latency_sec"])
        cell["gpu_edp_change_vs_default_pct"] = pct(cell["median_gpu_edp"], base["median_gpu_edp"])
        cell["latency_constraint_pass"] = cell["latency_change_vs_default_pct"] is not None and cell["latency_change_vs_default_pct"] <= 5.0
        cell["valid_edp_comparison"] = cell["status"] == base["status"] == "valid"


def cpu_effects(cells):
    output = []
    apps = sorted({k[1] for k in cells})
    for app in apps:
        for gov in GOVS:
            ps = cells[("cpu_powersave", app, "large", gov)]
            pf = cells[("cpu_performance", app, "large", gov)]
            valid = ps["status"] == pf["status"] == "valid"
            output.append({
                "benchmark": app, "size": "large", "governor": gov,
                "powersave_status": ps["status"], "performance_status": pf["status"],
                "powersave_latency_sec": ps["median_launch_latency_sec"],
                "performance_latency_sec": pf["median_launch_latency_sec"],
                "performance_vs_powersave_latency_pct": pct(pf["median_launch_latency_sec"], ps["median_launch_latency_sec"]),
                "powersave_gpu_edp": ps["median_gpu_edp"], "performance_gpu_edp": pf["median_gpu_edp"],
                "performance_vs_powersave_gpu_edp_pct": pct(pf["median_gpu_edp"], ps["median_gpu_edp"]) if valid else None,
                "valid_edp_pair": valid,
                "powersave_cpu_time_sec": ps["median_cpu_time_sec"], "performance_cpu_time_sec": pf["median_cpu_time_sec"],
                "performance_vs_powersave_cpu_time_pct": pct(pf["median_cpu_time_sec"], ps["median_cpu_time_sec"]),
            })
    return output


def recommendations(cells):
    output = []
    apps = sorted({k[1] for k in cells})
    for app in apps:
        candidates = []
        for gov in GOVS:
            cs = [cells[(cpu, app, "large", gov)] for cpu in INPUTS]
            bases = [cells[(cpu, app, "large", "gpu_default")] for cpu in INPUTS]
            valid = all(c["status"] == b["status"] == "valid" for c, b in zip(cs, bases))
            latency_ok = all(c["latency_change_vs_default_pct"] is not None and c["latency_change_vs_default_pct"] <= 5 for c in cs)
            if valid and latency_ok:
                ratios = [c["median_gpu_edp"] / b["median_gpu_edp"] for c, b in zip(cs, bases)]
                # A non-default policy is portable only if it improves GPU EDP
                # independently in both CPU modes. Default remains the safe
                # candidate at ratio 1.0.
                # Require at least a 1% EDP reduction independently in each
                # CPU mode before replacing the safe default. Smaller effects
                # remain documented as marginal observations, not static
                # runtime overrides.
                if gov == "gpu_default" or all(r <= 0.99 for r in ratios):
                    candidates.append((math.prod(ratios) ** 0.5, max(c["latency_change_vs_default_pct"] for c in cs), gov))
        statuses = {cpu: cells[(cpu, app, "large", "gpu_default")]["status"] for cpu in INPUTS}
        if candidates:
            score, worst_latency, gov = min(candidates)
            reason = "practical_cross_cpu_edp_improvement_ge_1pct_within_5pct_latency" if gov != "gpu_default" else "default_retained_no_practical_cross_cpu_win"
            portable = True
        elif all(v == "instrumentation_limited" for v in statuses.values()):
            gov, score, worst_latency = "gpu_default", None, 0.0
            reason = "default_retained_gpu_timer_zero_edp_unrankable"
            portable = False
        elif any(v == "censored_failure" for v in statuses.values()):
            gov, score, worst_latency = "gpu_default", None, 0.0
            reason = "default_retained_censored_gpu_measurement"
            portable = False
        else:
            gov, score, worst_latency = "gpu_default", 1.0, 0.0
            reason = "default_retained_no_cross_cpu_candidate"
            portable = True
        ps = cells[("cpu_powersave", app, "large", gov)]
        pf = cells[("cpu_performance", app, "large", gov)]
        output.append({
            "benchmark": app, "size": "large", "recommended_gpu_governor": gov, "reason": reason,
            "portable_across_cpu_modes": portable, "geomean_edp_change_vs_default_pct": None if score is None else (score - 1) * 100,
            "powersave_edp_change_vs_default_pct": ps["gpu_edp_change_vs_default_pct"],
            "performance_edp_change_vs_default_pct": pf["gpu_edp_change_vs_default_pct"],
            "powersave_latency_change_vs_default_pct": ps["latency_change_vs_default_pct"],
            "performance_latency_change_vs_default_pct": pf["latency_change_vs_default_pct"],
            "worst_latency_change_pct": worst_latency,
            "validation": "PASS" if portable and worst_latency <= 5 else "UNRANKABLE",
            "runtime_policy": "apply_static" if portable else "measure_gpu_timer_then_default",
        })
    return output


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def fonts():
    paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    bolds = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"]
    regular = next(p for p in paths if Path(p).exists())
    bold = next(p for p in bolds if Path(p).exists())
    return {str(s): ImageFont.truetype(regular, s) for s in (20, 24, 28, 32, 40, 52, 68)} | {f"b{s}": ImageFont.truetype(bold, s) for s in (20, 24, 28, 32, 40, 52, 68)}


F = fonts()


def canvas(title, subtitle=""):
    im = Image.new("RGB", (W, H), COLORS["paper"])
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 18), fill=COLORS["cyan"])
    d.text((80, 48), title, font=F["b40"], fill=COLORS["navy"])
    if subtitle:
        d.text((82, 104), subtitle, font=F["24"], fill=COLORS["muted"])
    return im, d


def save_plot(im, name):
    path = OUT / name
    im.save(path, "PNG", optimize=True)
    return path


def heat_color(value, lo, hi, reverse=False):
    if value is None:
        return "#b8bec5"
    t = 0.5 if hi == lo else max(0, min(1, (value - lo) / (hi - lo)))
    if reverse:
        t = 1 - t
    a, b = ((69, 143, 124), (210, 73, 91))
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def plot_matrix(cells, apps):
    im, d = canvas("Experimental matrix and measurement status", "Each tile is one 10-run cell; gray is censored, amber is zero GPU timer")
    x0, y0, cw, ch = 430, 180, 160, 25
    cols = [(cpu, gov) for cpu in INPUTS for gov in GOVS]
    for j, (cpu, gov) in enumerate(cols):
        x = x0 + j * cw
        d.text((x + 5, 145), f"{CPU_LABEL[cpu].split()[0]}\n{GOV_LABEL[gov]}", font=F["20"], fill=COLORS["ink"])
    status_color = {"valid": COLORS["cyan"], "censored_failure": "#aeb5bd", "instrumentation_limited": COLORS["amber"], "invalid_measurement": COLORS["red"]}
    for i, app in enumerate(apps):
        y = y0 + i * ch
        d.text((55, y + 2), app.replace("-cuda", ""), font=F["20"], fill=COLORS["ink"])
        for j, key in enumerate(cols):
            c = cells[(key[0], app, "large", key[1])]
            d.rectangle((x0 + j*cw, y, x0 + (j+1)*cw - 5, y+ch-3), fill=status_color[c["status"]])
    d.text((55, 980), "VALID", font=F["b20"], fill=COLORS["cyan"])
    d.text((240, 980), "CENSORED FAILURE", font=F["b20"], fill="#7c858f")
    d.text((555, 980), "ZERO-TIME / INSTRUMENTATION LIMITED", font=F["b20"], fill="#a46f00")
    return save_plot(im, "01_experimental_matrix.png")


def paired_dot_plot(rows, metric, title, subtitle, name, valid_only=False):
    im, d = canvas(title, subtitle)
    selected = [r for r in rows if r[metric] is not None and (not valid_only or r["valid_edp_pair"])]
    by_app = defaultdict(list)
    for r in selected:
        by_app[r["benchmark"]].append(r[metric])
    vals = [v for vs in by_app.values() for v in vs]
    cap = max(10, min(100, max(abs(v) for v in vals) * 1.1)) if vals else 10
    apps = sorted(by_app, key=lambda a: med(by_app[a]))
    x0, x1, y0, step = 490, 1700, 170, 27
    d.line((x0, 140, x0, 1010), fill=COLORS["grid"], width=3)
    for i, app in enumerate(apps):
        y = y0 + i * step
        d.text((55, y-10), app.replace("-cuda", ""), font=F["20"], fill=COLORS["ink"])
        for r in [z for z in selected if z["benchmark"] == app]:
            v = max(-cap, min(cap, r[metric])); x = x0 + (v + cap)/(2*cap)*(x1-x0)
            color = COLORS["blue"] if r["governor"] in ("gpu_default", "gpu_ondemand") else COLORS["purple"]
            d.ellipse((x-6, y-6, x+6, y+6), fill=color)
    zero = x0 + (x1-x0)/2
    d.line((zero, 140, zero, 1010), fill=COLORS["navy"], width=3)
    for tick in (-cap, -cap/2, 0, cap/2, cap):
        x = x0 + (tick+cap)/(2*cap)*(x1-x0)
        d.text((x-35, 1020), f"{tick:+.0f}%", font=F["20"], fill=COLORS["muted"])
    return save_plot(im, name)


def plot_edp_heatmaps(cells, apps):
    im, d = canvas("GPU governor EDP change vs default", "Median per-run GPU EDP; lower is better. Zero-time and censored cells are not ranked.")
    for panel, cpu in enumerate(INPUTS):
        px = 60 + panel * 875
        d.text((px, 145), CPU_LABEL[cpu], font=F["b28"], fill=COLORS["navy"])
        for j, gov in enumerate(GOVS[1:]):
            d.text((px+330+j*170, 180), GOV_LABEL[gov], font=F["20"], fill=COLORS["ink"])
        for i, app in enumerate(apps):
            y = 215 + i*27
            d.text((px, y+2), app.replace("-cuda", ""), font=F["20"], fill=COLORS["ink"])
            for j, gov in enumerate(GOVS[1:]):
                c = cells[(cpu, app, "large", gov)]
                v = c["gpu_edp_change_vs_default_pct"] if c["valid_edp_comparison"] else None
                x = px+325+j*170
                d.rectangle((x, y, x+158, y+24), fill=heat_color(v, -40, 40))
                d.text((x+38, y+1), signed(v, 1), font=F["20"], fill=COLORS["white"] if v is not None else COLORS["ink"])
    return save_plot(im, "04_gpu_governor_edp_heatmaps.png")


def plot_bottleneck(cells, apps):
    im, d = canvas("Time-component and bottleneck view", "Default governor medians; components are diagnostic and explicitly nonadditive")
    x0, x1, y0, step = 420, 1700, 175, 27
    vals = []
    for app in apps:
        c = cells[("cpu_performance", app, "large", "gpu_default")]
        vals.append(max(v or 0 for v in (c["median_cpu_time_sec"], c["median_gpu_time_sec"], c["median_memcpy_time_sec"])))
    vmax = max(vals)
    for i, app in enumerate(apps):
        c = cells[("cpu_performance", app, "large", "gpu_default")]
        y = y0+i*step
        d.text((55, y), app.replace("-cuda", ""), font=F["20"], fill=COLORS["ink"])
        parts = [(c["median_cpu_time_sec"], COLORS["blue"]), (c["median_gpu_time_sec"], COLORS["cyan"]), (c["median_memcpy_time_sec"], COLORS["amber"])]
        for k, (v, color) in enumerate(parts):
            width = 0 if v is None else math.log1p(v)/math.log1p(vmax)*(x1-x0)
            d.rectangle((x0, y+k*7, x0+width, y+k*7+5), fill=color)
    d.text((420, 1015), "CPU time", font=F["b20"], fill=COLORS["blue"])
    d.text((620, 1015), "GPU time", font=F["b20"], fill=COLORS["cyan"])
    d.text((820, 1015), "Memcpy", font=F["b20"], fill="#a46f00")
    d.text((1120, 1015), "Log-scaled seconds", font=F["20"], fill=COLORS["muted"])
    return save_plot(im, "05_time_components_bottlenecks.png")


def plot_wins(cells, apps):
    im, d = canvas("Valid GPU EDP wins under the latency guardrail", "Count of applications with lower EDP and <=5% application-latency change vs default")
    y = 280
    for cpu in INPUTS:
        d.text((80, y-55), CPU_LABEL[cpu], font=F["b28"], fill=COLORS["navy"])
        for j, gov in enumerate(GOVS[1:]):
            eligible = [cells[(cpu,a,"large",gov)] for a in apps]
            wins = sum(c["valid_edp_comparison"] and c["latency_constraint_pass"] and c["gpu_edp_change_vs_default_pct"] < 0 for c in eligible)
            losses = sum(c["valid_edp_comparison"] and c["latency_constraint_pass"] and c["gpu_edp_change_vs_default_pct"] >= 0 for c in eligible)
            unranked = len(apps)-wins-losses
            x = 350+j*430
            totalw = 330
            d.rectangle((x,y,x+totalw*wins/30,y+64), fill=COLORS["cyan"])
            d.rectangle((x+totalw*wins/30,y,x+totalw*(wins+losses)/30,y+64), fill=COLORS["red"])
            d.rectangle((x+totalw*(wins+losses)/30,y,x+totalw,y+64), fill="#aeb5bd")
            d.text((x,y-35), GOV_LABEL[gov], font=F["b24"], fill=COLORS["ink"])
            d.text((x,y+78), f"{wins} wins | {losses} non-wins | {unranked} unranked", font=F["20"], fill=COLORS["muted"])
        y += 360
    return save_plot(im, "06_valid_wins.png")


def plot_recommendations(recs):
    im, d = canvas("Policy recommendation and cross-CPU validation", "Recommended governor requires valid GPU EDP and <=5% latency in both CPU modes")
    apps = [r["benchmark"] for r in recs]
    x0, y0, step = 530, 175, 27
    for i, app in enumerate(apps):
        r = recs[i]; y = y0+i*step
        d.text((55,y),app.replace("-cuda",""),font=F["20"],fill=COLORS["ink"])
        d.text((350,y),GOV_LABEL[r["recommended_gpu_governor"]],font=F["b20"],fill=COLORS["navy"])
        if r["geomean_edp_change_vs_default_pct"] is None:
            d.text((x0,y),"UNRANKABLE",font=F["b20"],fill=COLORS["muted"])
        else:
            v=r["geomean_edp_change_vs_default_pct"]; width=min(480,abs(v)*8)
            color=COLORS["cyan"] if v<0 else COLORS["red"]
            d.rectangle((x0,y+2,x0+width,y+20),fill=color)
            d.text((1030,y),signed(v),font=F["20"],fill=COLORS["ink"])
        lat=max(r["powersave_latency_change_vs_default_pct"] or 0,r["performance_latency_change_vs_default_pct"] or 0)
        d.text((1250,y),f"worst latency {signed(lat)}",font=F["20"],fill=COLORS["green"] if lat<=5 else COLORS["red"])
    return save_plot(im, "07_policy_recommendation_validation.png")


def report_page(title, paragraphs, bullets=()):
    im, d = canvas(title)
    y = 155
    for p in paragraphs:
        for line in textwrap.wrap(p, width=105):
            d.text((90,y),line,font=F["24"],fill=COLORS["ink"]); y += 34
        y += 22
    for b in bullets:
        lines=textwrap.wrap(b,width=99)
        d.ellipse((92,y+10,102,y+20),fill=COLORS["cyan"])
        for j,line in enumerate(lines):
            d.text((122,y),line,font=F["24"],fill=COLORS["ink"]); y += 34
        y += 12
    d.text((90,1040),"DGX Spark paired governor analysis | generated by paired_analysis.py",font=F["20"],fill=COLORS["muted"])
    return im


def build_outputs(rows, cells, cpu_rows, recs, plots):
    cell_rows = [cells[k] for k in sorted(cells)]
    governor_fields = ["cpu_mode","benchmark","size","governor","n_runs","n_valid_edp","n_censored","n_zero_gpu_time","status",
        "median_gpu_edp","median_gpu_time_sec","median_power_draw_w","median_launch_latency_sec","median_cpu_time_sec","median_memcpy_time_sec",
        "gpu_edp_change_vs_default_pct","latency_change_vs_default_pct","latency_constraint_pass","valid_edp_comparison"]
    write_csv(OUT/"governor_effects.csv",cell_rows,governor_fields)
    cpu_fields=list(cpu_rows[0])
    write_csv(OUT/"cpu_mode_effects.csv",cpu_rows,cpu_fields)
    rec_fields=list(recs[0])
    write_csv(OUT/"recommendations.csv",recs,rec_fields)

    quality=[]
    for cpu,path in INPUTS.items():
        rr=[r for r in rows if r["cpu_mode"]==cpu]
        quality.append({"scope":cpu,"source_file":str(path),"runs":len(rr),"cells":len(rr)//10,
            "valid_edp_runs":sum(r["status"]=="valid" for r in rr),"censored_runs":sum(r["status"]=="censored_failure" for r in rr),
            "zero_gpu_time_runs":sum(r["status"]=="instrumentation_limited" for r in rr),
            "censored_cells":sum(c["cpu_mode"]==cpu and c["status"]=="censored_failure" for c in cell_rows),
            "instrumentation_limited_cells":sum(c["cpu_mode"]==cpu and c["status"]=="instrumentation_limited" for c in cell_rows),
            "experiment_first_ts":min(r["experiment_ts"] for r in rr),"experiment_last_ts":max(r["experiment_ts"] for r in rr),
            "treatment":"missing GPU time censored; zero GPU time excluded; EDP median of per-run products"})
    write_csv(OUT/"data_quality.csv",quality,list(quality[0]))

    valid_cpu=[r for r in cpu_rows if r["valid_edp_pair"]]
    lat_effects=[r["performance_vs_powersave_latency_pct"] for r in cpu_rows if r["performance_vs_powersave_latency_pct"] is not None]
    gpu_effects=[r["performance_vs_powersave_gpu_edp_pct"] for r in valid_cpu]
    cpu_time_effects=[r["performance_vs_powersave_cpu_time_pct"] for r in cpu_rows if r["performance_vs_powersave_cpu_time_pct"] is not None]
    censored=[f'{c["cpu_mode"]}:{c["benchmark"]}:{c["governor"]}' for c in cell_rows if c["status"]=="censored_failure"]
    limited=sorted({c["benchmark"] for c in cell_rows if c["status"]=="instrumentation_limited"})
    counts=Counter(r["recommended_gpu_governor"] for r in recs)
    wins=[r for r in recs if r["geomean_edp_change_vs_default_pct"] is not None and r["geomean_edp_change_vs_default_pct"]<0]
    best=sorted(wins,key=lambda r:r["geomean_edp_change_vs_default_pct"])

    finding = f"""# Paired DGX Spark CPU/GPU Governor Analysis

## Scope and method

This report pairs **2,400 runs**, **240 cells**, **30 applications**, four GPU governors, and two CPU modes. CPU powersave was measured at 858 MHz in the experiment beginning 20260716_163946 (file timestamps {quality[0]['experiment_first_ts']} to {quality[0]['experiment_last_ts']}); CPU performance was measured at 3354 MHz in experiment 20260720_191544 (file timestamps {quality[1]['experiment_first_ts']} to {quality[1]['experiment_last_ts']}). Every cell has 10 runs.

The metric is strictly `GPU_EDP = avg_gpu_time_sec^2 * power_draw_w_avg`, calculated per run and then aggregated as the median of the 10 per-run EDP values. `avg_launch_time_sec` is never used in EDP; it is application latency and supplies the <=5% guardrail. CPU EDP is omitted because DGX Spark CPU package power is unavailable. CPU time is analyzed only as latency.

## Exact aggregate effects

Across all 120 matched CPU-mode/governor/application pairs, changing CPU powersave to performance changed median application latency by **{signed(med(lat_effects),2)}** (median). Across the {len(valid_cpu)} valid GPU-EDP pairs, it changed GPU EDP by **{signed(med(gpu_effects),2)}** (median). Median CPU-time change was **{signed(med(cpu_time_effects),2)}**. These paired summaries show that CPU governor effects on application latency and GPU EDP are distinct and must not be conflated.

Robust recommendations are portable only when the same GPU governor has valid EDP in both CPU modes and stays within +5% application latency in both. **{len(wins)} of 30 applications** obtain a lower cross-CPU geometric-mean EDP than default. Recommendation counts are: {', '.join(f'{GOV_LABEL[k]} {v}' for k,v in sorted(counts.items()))}. Top robust reductions are: {', '.join(f'{r["benchmark"]} {signed(r["geomean_edp_change_vs_default_pct"],2)} via {GOV_LABEL[r["recommended_gpu_governor"]]}' for r in best[:10]) or 'none'}.

## Application recommendations

{os.linesep.join(f'- **{r["benchmark"]}**: {GOV_LABEL[r["recommended_gpu_governor"]]}; EDP {signed(r["geomean_edp_change_vs_default_pct"],2)}, powersave latency {signed(r["powersave_latency_change_vs_default_pct"],2)}, performance latency {signed(r["performance_latency_change_vs_default_pct"],2)}; {r["validation"]}.' for r in recs)}

## Censoring and instrumentation

Missing/`None` GPU time is a censored failure, never zero energy. The {len(censored)} censored cells are: {', '.join(censored)}. Any governor comparison involving these cells is unrankable.

Zero GPU time is an instrumentation limitation and is excluded from all EDP percentage rankings. Affected applications are **{', '.join(limited)}**. This explicitly includes `gemmEx-cuda` and `memcpy-cuda`; no zero-time observation is represented as a zero-energy win.

## Portable runtime policy

Use the static application-to-governor map in `recommendations.csv` only for rows marked `PASS`. At runtime, verify a positive GPU timer, completed launches, power sample count, GPU utilization, clocks, temperature/throttling state, problem size, and an application-latency sentinel. Fall back to `gpu_default` if GPU time is missing/zero, the workload is unknown, the latency sentinel exceeds +5%, or counters leave the calibration envelope. Periodically re-profile because power/thermal state and software versions can shift the optimum.

## Limitations

- The CPU-mode experiments occurred on different dates, so CPU mode is confounded with date, software state, thermal history, and other uncontrolled drift. Pairing by application/governor does not remove that confounding.
- CPU, GPU, memcpy, and launch timers may overlap and are nonadditive. The component plot is a bottleneck diagnostic, not a decomposition of launch latency.
- CPU package power is absent on DGX Spark, so CPU EDP and whole-system EDP cannot be computed. CPU time and launch time are latency metrics only.
- Median cell estimates summarize 10 runs but do not establish statistical significance or causal effects.
- Results cover the measured large input and this hardware/software environment; portability requires runtime validation.
"""
    (OUT/"FINDINGS.md").write_text(finding,encoding="utf-8")

    pages=[report_page("DGX Spark paired governor analysis",[
        "A controlled comparison of four GPU governors under CPU powersave (858 MHz) and CPU performance (3354 MHz).",
        "Primary metric: median of per-run GPU EDP = avg_gpu_time_sec squared times power_draw_w_avg. Application latency is avg_launch_time_sec and is only the <=5% constraint."],
        [f"2,400 runs | 240 ten-run cells | 30 applications | {len(valid_cpu)} valid paired CPU-mode EDP comparisons",
         f"Median performance-vs-powersave effect: application latency {signed(med(lat_effects),2)}; GPU EDP {signed(med(gpu_effects),2)}.",
         "Missing GPU time is censored. Zero GPU time is instrumentation-limited. Neither is zero energy."]),
        report_page("Method and decision rule",["For each run, calculate GPU EDP from GPU time and average board power. For each 10-run cell, take the median of those per-run products. Compare each non-default GPU governor with default inside the same CPU mode.",
        "A robust recommendation must be EDP-valid and have application latency no more than 5% above default in both CPU modes. The selected candidate minimizes the geometric mean EDP ratio across CPU modes."],
        ["avg_launch_time_sec is never an EDP input.","CPU EDP is omitted because CPU package power is unavailable.","All percentage rankings exclude censored and zero-GPU-time cells."])]
    for p in plots:
        pages.append(Image.open(p).convert("RGB"))
    pages += [report_page("Findings and deployment",[
        f"CPU performance changed median application latency by {signed(med(lat_effects),2)} and valid-pair GPU EDP by {signed(med(gpu_effects),2)} versus CPU powersave. These are separate responses: CPU frequency can alter host-side latency without a proportional GPU EDP response.",
        f"{len(wins)} applications have a robust lower-EDP recommendation. Full per-application values and validation are in recommendations.csv and FINDINGS.md."],
        ["Deploy only PASS mappings; preserve default for unrankable timers.","Validate positive GPU time, power samples, completed launches, utilization, clocks, thermals, problem size, and the +5% latency sentinel.","Re-profile after driver, runtime, benchmark, thermal, or hardware changes."]),
        report_page("Limitations",[],["CPU modes were measured on different dates; date and CPU mode are confounded.","Component timers overlap and are nonadditive.","CPU package power is unavailable, preventing CPU and whole-system EDP.","Ten-run medians are descriptive and do not imply statistical significance.","Conclusions apply to the measured large-input DGX Spark environment."])]
    pages[0].save(OUT/"analysis_report.pdf","PDF",save_all=True,append_images=pages[1:],resolution=150.0,title="Paired DGX Spark Governor Analysis",author="paired_analysis.py")


def validate(rows, cells, cpu_rows, recs, plots):
    assert len(rows)==2400 and len(cells)==240 and len(cpu_rows)==120 and len(recs)==30
    assert all(c["n_runs"]==10 for c in cells.values())
    # Recompute every valid EDP aggregate independently from source rows.
    bykey=defaultdict(list)
    for r in rows:
        bykey[(r["cpu_mode"],r["benchmark"],r["size"],r["governor"])].append(r)
    for key,c in cells.items():
        raw=bykey[key]
        expected=med([r["gpu_time"]**2*r["power"] for r in raw if r["status"]=="valid"])
        if c["status"]=="valid":
            assert math.isclose(c["median_gpu_edp"],expected,rel_tol=1e-12)
        else:
            assert c["median_gpu_edp"] is None
    assert all(r["worst_latency_change_pct"]<=5+1e-9 for r in recs if r["validation"]=="PASS")
    expected=["paired_analysis.py","analysis_report.pdf","governor_effects.csv","cpu_mode_effects.csv","recommendations.csv","data_quality.csv","FINDINGS.md"]+[p.name for p in plots]
    for name in expected:
        path=OUT/name
        assert path.is_file() and path.stat().st_size>100, f"Invalid output {path}"
    for p in plots:
        with Image.open(p) as im:
            assert im.size==(W,H) and im.format=="PNG"
    assert (OUT/"analysis_report.pdf").read_bytes()[:5]==b"%PDF-"
    (OUT/"VALIDATION.txt").write_text(
        f"PASS {datetime.now().isoformat(timespec='seconds')}\n"
        f"Validated 2400 runs, 240 cells, 120 CPU pairs, 30 recommendations, {len(plots)} PNG plots.\n"
        "Confirmed GPU EDP = avg_gpu_time_sec^2 * power_draw_w_avg per run; cell median is median of per-run EDP.\n"
        "Confirmed nonvalid cells have no EDP, recommendation latency guardrails, CSVs, PNG dimensions, and PDF signature.\n",encoding="utf-8")


def main():
    rows=load_data(); cells=aggregate(rows); enrich(cells)
    cpus=cpu_effects(cells); recs=recommendations(cells)
    apps=sorted({r["benchmark"] for r in rows})
    plots=[plot_matrix(cells,apps),
        paired_dot_plot(cpus,"performance_vs_powersave_latency_pct","CPU governor impact on application latency","Performance vs powersave; each dot is one GPU-governor pairing","02_cpu_impact_latency.png"),
        paired_dot_plot(cpus,"performance_vs_powersave_gpu_edp_pct","CPU governor impact on GPU EDP","Performance vs powersave; valid positive GPU timers only","03_cpu_impact_gpu_edp.png",True),
        plot_edp_heatmaps(cells,apps),plot_bottleneck(cells,apps),plot_wins(cells,apps),plot_recommendations(recs)]
    build_outputs(rows,cells,cpus,recs,plots)
    validate(rows,cells,cpus,recs,plots)
    print(f"Generated and validated {len(plots)} plots and report in {OUT}")


if __name__=="__main__":
    main()

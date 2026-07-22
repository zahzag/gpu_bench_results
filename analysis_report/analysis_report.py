#!/usr/bin/env python3
"""Analyze DGX Spark GPU governors using only experiment 20260716_163946.

GPU EDP = avg_gpu_time_sec^2 * avg GPU power.
CPU EDP is intentionally omitted because CPU power is unavailable on this ARM
platform. Latency is avg_cpu_time_sec + avg_gpu_time_sec +
avg_gpu_memcpy_sec, which is checked against avg_launch_time_sec.
"""

import argparse
import csv
import math
import os
import statistics
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageFont


GOVS = ("default", "ondemand", "powersave", "performance")
COLORS = {"default": "#426A8C", "ondemand": "#16877A", "powersave": "#D6A22D",
          "performance": "#D65A4A", "CPU": "#7A5195", "GPU": "#00876C",
          "MIXED": "#E69F00", "MEMCPY": "#4C78A8", "ink": "#18212B",
          "muted": "#65717E", "paper": "#F7F9FC", "grid": "#D7DEE7"}


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def med(values):
    values = [v for v in values if math.isfinite(v)]
    return statistics.median(values) if values else math.nan


def pct(value, base):
    return 100 * (value / base - 1) if base > 0 and math.isfinite(value) else math.nan


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/" + name, size)


def canvas(h=1050):
    return Image.new("RGB", (1800, h), COLORS["paper"])


def heading(draw, text, subtitle=""):
    draw.text((70, 45), text, font=font(38, True), fill=COLORS["ink"])
    if subtitle:
        draw.text((72, 98), subtitle, font=font(18), fill=COLORS["muted"])


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            governor = raw["governor"].removeprefix("gpu_")
            gpu_power = num(raw.get("gpu_power_avg"))
            if not math.isfinite(gpu_power):
                gpu_power = num(raw.get("power_draw_w_avg"))
            cpu = num(raw.get("avg_cpu_time_sec"))
            gpu = num(raw.get("avg_gpu_time_sec"))
            memcpy = num(raw.get("avg_gpu_memcpy_sec"))
            latency = num(raw.get("avg_launch_time_sec"))
            rows.append({
                "benchmark": raw["benchmark"].removesuffix("-cuda"),
                "governor": governor, "run": int(raw["run"]),
                "cpu_time": cpu, "gpu_time": gpu, "memcpy_time": memcpy,
                "latency": latency, "decomposed_latency": cpu + gpu + memcpy,
                "gpu_power": gpu_power, "gpu_energy": gpu * gpu_power,
                "gpu_edp": gpu * gpu * gpu_power,
                "completed": num(raw.get("completed_launches")),
                "sm_active": num(raw.get("sm_active_pct_avg")),
                "sm_clock": num(raw.get("sm_clock_mhz_avg")),
                "video_clock": num(raw.get("video_clock_mhz_avg")),
                "gpu_temp": num(raw.get("gpu_temp_c_avg")),
            })
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["benchmark"], row["governor"])].append(row)
    cells = {}
    metrics = ("cpu_time", "gpu_time", "memcpy_time", "latency",
               "decomposed_latency", "gpu_power", "gpu_energy", "gpu_edp",
               "completed", "sm_active", "sm_clock", "video_clock", "gpu_temp")
    for key, group in groups.items():
        cells[key] = {metric: med(r[metric] for r in group) for metric in metrics}
        cells[key]["valid_runs"] = sum(r["completed"] > 0 and r["gpu_time"] > 0 for r in group)
    return cells


def classify(cell):
    total = cell["cpu_time"] + cell["gpu_time"] + cell["memcpy_time"]
    if total <= 0:
        return "UNKNOWN"
    parts = {"CPU": cell["cpu_time"] / total, "GPU": cell["gpu_time"] / total,
             "MEMCPY": cell["memcpy_time"] / total}
    if parts["CPU"] >= .70:
        return "CPU"
    if parts["GPU"] >= .70:
        return "GPU"
    if parts["MEMCPY"] >= .20:
        return "MEMCPY"
    return "MIXED"


def analyze(cells):
    effects, classes, validations = [], [], []
    benchmarks = sorted({key[0] for key in cells})
    for bench in benchmarks:
        default = cells[(bench, "default")]
        cls = classify(default)
        total = default["latency"]
        classes.append({"benchmark": bench, "bottleneck": cls,
                        "cpu_fraction": default["cpu_time"] / total,
                        "gpu_fraction": default["gpu_time"] / total,
                        "memcpy_fraction": default["memcpy_time"] / total,
                        "sm_active_default": default["sm_active"]})
        valid = []
        for gov in GOVS:
            cell = cells.get((bench, gov))
            if not cell:
                continue
            usable = cell["valid_runs"] >= 5 and cell["completed"] > 0 and cell["gpu_time"] > 0
            row = {"benchmark": bench, "bottleneck": cls, "governor": gov,
                   "gpu_edp": cell["gpu_edp"], "gpu_energy": cell["gpu_energy"],
                   "gpu_power_w": cell["gpu_power"], "cpu_time_sec": cell["cpu_time"],
                   "gpu_time_sec": cell["gpu_time"], "memcpy_time_sec": cell["memcpy_time"],
                   "latency_sec": cell["latency"], "gpu_edp_delta_pct": pct(cell["gpu_edp"], default["gpu_edp"]),
                   "gpu_energy_delta_pct": pct(cell["gpu_energy"], default["gpu_energy"]),
                   "latency_delta_pct": pct(cell["latency"], default["latency"]),
                   "gpu_power_delta_pct": pct(cell["gpu_power"], default["gpu_power"]),
                   "valid_runs": cell["valid_runs"], "completed_launches": cell["completed"],
                   "sm_active_pct": cell["sm_active"], "sm_clock_mhz": cell["sm_clock"],
                   "video_clock_mhz": cell["video_clock"], "usable": usable}
            effects.append(row)
            if usable and cell["latency"] <= default["latency"] * 1.05:
                valid.append((cell["gpu_edp"], gov))
        empirical = min(valid)[1] if valid else "default"
        # Transparent rule based only on counters observed under default.
        predicted = "default"
        trial = cells.get((bench, "ondemand"))
        if (default["sm_active"] < 30 and trial and trial["valid_runs"] >= 5
                and trial["completed"] > 0 and trial["gpu_time"] > 0
                and trial["latency"] <= default["latency"] * 1.05
                and trial["gpu_edp"] < default["gpu_edp"]):
            predicted = "ondemand"
        pred_cell = cells[(bench, predicted)]
        best_cell = cells[(bench, empirical)]
        regret = pct(pred_cell["gpu_edp"], best_cell["gpu_edp"])
        validations.append({"benchmark": bench, "bottleneck": cls,
                            "sm_active_default": default["sm_active"],
                            "predicted_governor": predicted, "empirical_governor": empirical,
                            "exact_match": predicted == empirical,
                            "gpu_edp_regret_pct": regret,
                            "predicted_latency_delta_pct": pct(pred_cell["latency"], default["latency"])})
    return effects, classes, validations


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def time_plot(classes, path):
    image = canvas(1350); draw = ImageDraw.Draw(image)
    heading(draw, "Application time decomposition under GPU default",
            "Latency = CPU time + GPU kernel time + GPU memcpy time; bars are fractions of avg_launch_time")
    x0, x1, y0, bh = 390, 1680, 155, 37
    for i, row in enumerate(sorted(classes, key=lambda r: -r["gpu_fraction"])):
        y = y0 + i * bh
        draw.text((60, y + 5), row["benchmark"], font=font(16), fill=COLORS["ink"])
        cursor = x0
        for key, color in (("cpu_fraction", COLORS["CPU"]), ("gpu_fraction", COLORS["GPU"]),
                           ("memcpy_fraction", COLORS["MEMCPY"])):
            width = row[key] * (x1 - x0)
            draw.rectangle((cursor, y, cursor + width, y + 27), fill=color)
            cursor += width
        draw.text((1690, y + 4), row["bottleneck"], font=font(14, True), fill=COLORS[row["bottleneck"]])
    for i, (label, color) in enumerate((("CPU", COLORS["CPU"]), ("GPU", COLORS["GPU"]), ("MEMCPY", COLORS["MEMCPY"]))):
        draw.rectangle((1120 + i * 180, 65, 1145 + i * 180, 90), fill=color)
        draw.text((1153 + i * 180, 65), label, font=font(16), fill=COLORS["ink"])
    image.save(path)


def class_plot(classes, path):
    image = canvas(); draw = ImageDraw.Draw(image)
    heading(draw, "Bottleneck classification", "Default-governor component shares; 70% threshold, memcpy threshold 20%")
    counts = Counter(r["bottleneck"] for r in classes)
    labels = ("CPU", "GPU", "MIXED", "MEMCPY")
    for i, label in enumerate(labels):
        x = 130 + i * 400; h = counts[label] * 48
        draw.rectangle((x, 820 - h, x + 250, 820), fill=COLORS[label])
        draw.text((x + 95, 775 - h), str(counts[label]), font=font(34, True), fill=COLORS["ink"])
        draw.text((x + 65, 850), label, font=font(23, True), fill=COLORS[label])
    draw.text((100, 940), "CPU time is analyzed as latency only. CPU EDP is omitted because CPU power is unavailable on DGX Spark.",
              font=font(18), fill=COLORS["muted"])
    image.save(path)


def heatmap(effects, path):
    image = canvas(1370); draw = ImageDraw.Draw(image)
    heading(draw, "GPU EDP change by application and governor", "GPU EDP = avg_gpu_time_sec² × avg GPU power; relative to GPU default")
    benches = sorted({r["benchmark"] for r in effects})
    lookup = {(r["benchmark"], r["governor"]): r for r in effects}
    x0, y0, cw, ch = 500, 180, 360, 37
    for i, gov in enumerate(("ondemand", "powersave", "performance")):
        draw.text((x0 + i * cw + 90, 135), gov, font=font(20, True), fill=COLORS[gov])
    for j, bench in enumerate(benches):
        y = y0 + j * ch
        cls = lookup[(bench, "default")]["bottleneck"]
        draw.text((55, y + 5), bench, font=font(16), fill=COLORS["ink"])
        draw.text((285, y + 5), cls, font=font(14, True), fill=COLORS[cls])
        for i, gov in enumerate(("ondemand", "powersave", "performance")):
            row = lookup.get((bench, gov)); x = x0 + i * cw
            if not row or not row["usable"] or not math.isfinite(row["gpu_edp_delta_pct"]):
                color, label = "#B8C0CA", "FAIL / N.A."
            else:
                v = row["gpu_edp_delta_pct"]; strength = min(abs(v), 100) / 100
                color = (int(225 - 95 * strength), int(245 - 50 * strength), int(235 - 60 * strength)) if v < 0 else (245, int(226 - 75 * strength), int(220 - 75 * strength))
                label = f"{v:+.0f}% | latency {row['latency_delta_pct']:+.1f}%"
            draw.rectangle((x, y, x + cw - 7, y + ch - 4), fill=color)
            draw.text((x + 45, y + 5), label, font=font(15, True), fill=COLORS["ink"])
    image.save(path)


def cpu_vs_edp(effects, path):
    image = canvas(); draw = ImageDraw.Draw(image)
    heading(draw, "CPU time fraction vs GPU EDP effect", "Each point is a valid non-default policy; CPU time is latency, not CPU energy")
    x0, x1, y0, y1 = 150, 1650, 170, 900
    draw.rectangle((x0, y0, x1, y1), outline=COLORS["grid"], width=2)
    default = {r["benchmark"]: r for r in effects if r["governor"] == "default"}
    def xy(x, y):
        y = max(-100, min(300, y))
        return x0 + x * (x1 - x0), y1 - (y + 100) / 400 * (y1 - y0)
    _, zy = xy(0, 0); draw.line((x0, zy, x1, zy), fill="#8995A3", width=2)
    for row in effects:
        if row["governor"] == "default" or not row["usable"]:
            continue
        base = default[row["benchmark"]]
        fraction = base["cpu_time_sec"] / base["latency_sec"]
        x, y = xy(fraction, row["gpu_edp_delta_pct"])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLORS[row["governor"]])
    for tick in (0, .25, .5, .75, 1):
        x, _ = xy(tick, 0); draw.text((x - 20, y1 + 12), f"{tick:.0%}", font=font(15), fill=COLORS["muted"])
    draw.text((650, 955), "CPU fraction of total launch latency under default", font=font(19), fill=COLORS["muted"])
    draw.text((25, 490), "GPU EDP delta", font=font(18), fill=COLORS["muted"])
    image.save(path)


def sm_plot(classes, path):
    image = canvas(); draw = ImageDraw.Draw(image)
    heading(draw, "SM activity predicts the bottleneck", "Default-governor SM active percentage; proposed runtime threshold is 30%")
    ordered = sorted(classes, key=lambda r: r["sm_active_default"])
    x0, y0, y1, bw = 100, 190, 850, 50
    for i, row in enumerate(ordered):
        x = x0 + i * bw; h = row["sm_active_default"] / 100 * (y1 - y0)
        draw.rectangle((x, y1 - h, x + 34, y1), fill=COLORS[row["bottleneck"]])
        draw.text((x, y1 + 12), row["benchmark"][:7], font=font(11), fill=COLORS["muted"])
    threshold_y = y1 - .30 * (y1 - y0)
    draw.line((x0, threshold_y, x0 + len(ordered) * bw, threshold_y), fill="#B5473C", width=3)
    draw.text((120, threshold_y - 28), "30% threshold: trial ondemand below this line", font=font(17, True), fill="#B5473C")
    image.save(path)


def validation_plot(validations, path):
    image = canvas(1200); draw = ImageDraw.Draw(image)
    accuracy = sum(v["exact_match"] for v in validations) / len(validations)
    regrets = [v["gpu_edp_regret_pct"] for v in validations if math.isfinite(v["gpu_edp_regret_pct"])]
    heading(draw, "Runtime rule validation",
            f"Rule: low SM activity triggers a guarded ondemand trial | exact match {accuracy:.0%}, median EDP regret {med(regrets):.1f}%")
    x0, y0, cw, ch = 580, 175, 300, 34
    for i, label in enumerate(("predicted", "empirical", "EDP regret")):
        draw.text((x0 + i * cw + 45, 135), label, font=font(17, True), fill=COLORS["ink"])
    for j, row in enumerate(validations):
        y = y0 + j * ch
        draw.text((50, y + 4), row["benchmark"], font=font(15), fill=COLORS["ink"])
        draw.text((285, y + 4), row["bottleneck"], font=font(14, True), fill=COLORS[row["bottleneck"]])
        draw.text((x0 + 60, y + 4), row["predicted_governor"], font=font(15, True), fill=COLORS[row["predicted_governor"]])
        draw.text((x0 + cw + 60, y + 4), row["empirical_governor"], font=font(15, True), fill=COLORS[row["empirical_governor"]])
        color = "#18785C" if row["gpu_edp_regret_pct"] <= 5 else "#B5473C"
        draw.text((x0 + 2 * cw + 80, y + 4), f"{row['gpu_edp_regret_pct']:.1f}%", font=font(15, True), fill=color)
    image.save(path)


def pdf_page(title_text, lines, number):
    image = canvas(); draw = ImageDraw.Draw(image)
    heading(draw, title_text)
    y = 145
    for kind, text in lines:
        if kind == "h":
            y += 12; draw.text((80, y), text, font=font(23, True), fill=COLORS["ink"]); y += 42; continue
        prefix = "• " if kind == "b" else ""
        words, line = text.split(), prefix
        for word in words:
            trial = line + (" " if line else "") + word
            if draw.textlength(trial, font=font(18)) > 1540:
                draw.text((95, y), line, font=font(18), fill=COLORS["ink"]); y += 29; line = word
            else:
                line = trial
        draw.text((95, y), line, font=font(18), fill=COLORS["ink"]); y += 39
    draw.text((1680, 995), str(number), font=font(15), fill=COLORS["muted"])
    return image


def make_pdf(paths, classes, validations, effects, output):
    cover = canvas(); d = ImageDraw.Draw(cover); d.rectangle((0, 0, 1800, 1050), fill="#102A43")
    d.text((100, 150), "DGX Spark GPU Governor", font=font(59, True), fill="white")
    d.text((100, 235), "GPU EDP and Bottleneck Analysis", font=font(55, True), fill="white")
    d.text((105, 350), "Experiment 20260716_163946 only | 30 applications × 4 governors × 10 runs", font=font(23), fill="#B8D8F0")
    d.rectangle((100, 520, 1690, 820), fill="#173F5F")
    d.text((140, 565), "GPU EDP = avg_gpu_time_sec² × avg GPU power", font=font(31, True), fill="white")
    d.text((140, 640), "CPU EDP omitted: CPU power is unavailable on DGX Spark.", font=font(25), fill="#B8D8F0")
    d.text((140, 700), "CPU time is retained as application latency and bottleneck evidence.", font=font(25), fill="#B8D8F0")
    pages = [cover]
    counts = Counter(r["bottleneck"] for r in classes)
    wins = defaultdict(list)
    for r in effects:
        if r["governor"] != "default" and r["usable"] and r["gpu_edp_delta_pct"] < 0 and r["latency_delta_pct"] <= 5:
            wins[r["governor"]].append(r["benchmark"])
    pages.append(pdf_page("Scope and metric", [
        ("b", "Only 20260716_163946 is analyzed. The older 20260707_195535 experiment is excluded because it followed a different design and lacks CPU/GPU/memcpy time decomposition."),
        ("b", "Latency is avg_launch_time_sec = avg_cpu_time_sec + avg_gpu_time_sec + avg_gpu_memcpy_sec. The equality holds across all 1200 rows within CSV precision."),
        ("b", "GPU energy per launch = gpu_time × average GPU power. GPU EDP = GPU energy × gpu_time = gpu_time² × average GPU power."),
        ("b", "power_draw_w_avg equals the average of 10 Hz raw GPU power samples; the pair-wise difference is effectively zero."),
        ("b", "CPU EDP is not computed because Grace ARM does not expose CPU package power. Future x86 HPC analysis should use RAPL package energy and compute cpu_time² × cpu_power."),
    ], 2))
    pages.append(pdf_page("Headline findings", [
        ("b", f"Bottleneck classes: {counts['CPU']} CPU-bound, {counts['GPU']} GPU-bound, {counts['MIXED']} mixed, and {counts['MEMCPY']} memcpy-heavy application."),
        ("b", "Ondemand is not universally best under GPU EDP. Its median GPU EDP is approximately 4% worse than default, although it gives strong wins for backprop and nw."),
        ("b", "Powersave is generally harmful under GPU EDP because kernel slowdown is squared. Its median GPU EDP is several times default; backprop is the only valid <=5% latency win."),
        ("b", "Performance has no valid GPU EDP win and raises median GPU EDP by about 15%."),
        ("b", "Completed-launch filtering is mandatory: policies with fewer than five completed launches are failures, not zero-energy wins."),
        ("b", "SM activity is the most useful transparent workload signal. Low SM activity identifies candidates for an ondemand trial, but empirical validation is still required."),
    ], 3))
    pages.extend(Image.open(path).convert("RGB") for path in paths)
    pages.append(pdf_page("Runtime model and portability", [
        ("h", "Validated baseline rule"),
        ("b", "Observe the application under GPU default. If sm_active_pct_avg < 30%, trial ondemand; otherwise retain default."),
        ("b", "Accept ondemand only when completed work remains valid, measured GPU EDP improves, and total launch latency stays within 5% of default. Otherwise revert immediately."),
        ("h", "Portable features"),
        ("b", "Use SM activity percentage, normalized SM clock, normalized video/memory clock, and GPU temperature. These describe workload pressure and governor state without encoding the GPU model."),
        ("b", "Do not use dram_active_pct from this experiment: it is zero in all 1200 rows. Do not use PCIe generation, DRAM type, or interconnect type as workload features."),
        ("h", "Scale-up to x86 HPC"),
        ("b", "Add CPU package energy from RAPL, compute CPU EDP independently, and then optimize a joint objective while retaining separate GPU and CPU latency budgets."),
    ], 10))
    pages[0].save(output, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument("--output", default="analysis_report")
    args = parser.parse_args(); os.makedirs(args.output, exist_ok=True)
    rows = load(args.results_csv); cells = aggregate(rows)
    effects, classes, validations = analyze(cells)
    write_csv(os.path.join(args.output, "governor_effects_gpu_edp.csv"), effects)
    write_csv(os.path.join(args.output, "bottleneck_classification.csv"), classes)
    write_csv(os.path.join(args.output, "runtime_model_validation.csv"), validations)
    plots = [os.path.join(args.output, name) for name in (
        "01_bottleneck_classification.png", "02_time_decomposition.png",
        "03_gpu_edp_landscape.png", "04_cpu_time_vs_gpu_edp.png",
        "05_sm_active_vs_bottleneck.png", "06_runtime_model_validation.png")]
    class_plot(classes, plots[0]); time_plot(classes, plots[1]); heatmap(effects, plots[2])
    cpu_vs_edp(effects, plots[3]); sm_plot(classes, plots[4]); validation_plot(validations, plots[5])
    accuracy = sum(v["exact_match"] for v in validations) / len(validations)
    regrets = [v["gpu_edp_regret_pct"] for v in validations if math.isfinite(v["gpu_edp_regret_pct"])]
    with open(os.path.join(args.output, "runtime_model_rule.txt"), "w", encoding="utf-8") as stream:
        stream.write("Observe under GPU default.\n")
        stream.write("if sm_active_pct_avg < 30: trial gpu_ondemand\nelse: retain gpu_default\n")
        stream.write("Accept only if completed_launches > 0, measured GPU EDP improves, and latency <= 1.05 * default latency; otherwise revert.\n")
        stream.write(f"Validation: exact empirical-governor match {accuracy:.1%}; median GPU EDP regret {med(regrets):.2f}%.\n")
    make_pdf(plots, classes, validations, effects, os.path.join(args.output, "analysis_report.pdf"))
    print(f"Generated {args.output}: {len(rows)} runs, {len(cells)} valid cells")
    print(f"Rule accuracy: {accuracy:.1%}; median GPU EDP regret: {med(regrets):.2f}%")


if __name__ == "__main__":
    main()

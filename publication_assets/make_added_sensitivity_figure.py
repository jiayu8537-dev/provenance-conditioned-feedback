#!/usr/bin/env python3
"""Recreate the two-panel added-sensitivity figure from archived endpoint tables."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables" / "added_sensitivity"
OUT = ROOT / "publication_assets" / "generated"
SOURCE_OUT = OUT / "source_data"

COLORS = {
    "moderate_asymmetric": "#D97706",
    "strong_premium_penalty": "#B22222",
    "ai_appreciation": "#008577",
}
LABELS = {
    "moderate_asymmetric": "Moderate asymmetric response",
    "strong_premium_penalty": "Strong premium–penalty",
    "ai_appreciation": "AI appreciation",
}
MARKERS = {
    "moderate_asymmetric": "square",
    "strong_premium_penalty": "circle",
    "ai_appreciation": "triangle",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def register_fonts() -> tuple[str, str]:
    regular = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("FigureSans", regular))
        pdfmetrics.registerFont(TTFont("FigureSans-Bold", bold))
        return "FigureSans", "FigureSans-Bold"
    return "Helvetica", "Helvetica-Bold"


def hex_color(value: str):
    from reportlab.lib.colors import HexColor

    return HexColor(value)


def marker(c: canvas.Canvas, kind: str, x: float, y: float, color, size: float = 3.5) -> None:
    c.setFillColor(color)
    c.setStrokeColorRGB(1, 1, 1)
    c.setLineWidth(0.55)
    if kind == "circle":
        c.circle(x, y, size, stroke=1, fill=1)
    elif kind == "square":
        c.rect(x - size, y - size, size * 2, size * 2, stroke=1, fill=1)
    else:
        path = c.beginPath()
        path.moveTo(x, y - size * 1.15)
        path.lineTo(x - size * 1.1, y + size * 0.85)
        path.lineTo(x + size * 1.1, y + size * 0.85)
        path.close()
        c.drawPath(path, stroke=1, fill=1)


def error_bar_vertical(c, x, y, low, high, transform_y, color) -> None:
    y0, y1 = transform_y(low), transform_y(high)
    c.setStrokeColor(color)
    c.setLineWidth(0.9)
    c.line(x, y0, x, y1)
    c.line(x - 2.2, y0, x + 2.2, y0)
    c.line(x - 2.2, y1, x + 2.2, y1)


def error_bar_horizontal(c, y, low, high, transform_x, color) -> None:
    x0, x1 = transform_x(low), transform_x(high)
    c.setStrokeColor(color)
    c.setLineWidth(0.9)
    c.line(x0, y, x1, y)
    c.line(x0, y - 2.2, x0, y + 2.2)
    c.line(x1, y - 2.2, x1, y + 2.2)


def main() -> None:
    font, bold = register_fonts()
    low_all = rows(TABLES / "lower_acceptance_endpoints.csv")
    paired_all = rows(TABLES / "dynamic_minus_fixed_paired_LMC.csv")
    order = ["moderate_asymmetric", "strong_premium_penalty", "ai_appreciation"]
    low = [r for r in low_all if r["metric"] == "LMC" and r["scenario"] in order]
    paired = [r for r in paired_all if r["scenario"] in order]

    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TABLES / "lower_acceptance_endpoints.csv", SOURCE_OUT / "FigS_added_sensitivity_panel_a.csv")
    shutil.copy2(TABLES / "dynamic_minus_fixed_paired_LMC.csv", SOURCE_OUT / "FigS_added_sensitivity_panel_b.csv")

    width, height = 515.0, 248.0
    pdf_path = OUT / "FigS_added_sensitivity.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(width, height))
    c.setTitle("Added sensitivity analyses")
    c.setAuthor("")

    # Layout.
    left_x, left_y, left_w, plot_h = 58.0, 41.0, 244.0, 145.0
    right_x, right_y, right_w = 353.0, 41.0, 146.0
    grid = hex_color("#D9DDE3")
    axis = hex_color("#2E3338")
    zero = hex_color("#5F6670")

    c.setFillColor(axis)
    c.setFont(bold, 9.3)
    c.drawString(18, 202, "(a)")
    c.setFont(font, 9.0)
    c.drawString(42, 202, "Lower-feedback conditions")
    c.setFont(bold, 9.3)
    c.drawString(318, 202, "(b)")
    c.setFont(font, 9.0)
    c.drawString(342, 202, "Candidate-pool refresh")

    # Shared legend above panel a.
    legend_y = 224.0
    legend_x = [58.0, 170.0, 272.0]
    for x, scenario in zip(legend_x, order):
        color = hex_color(COLORS[scenario])
        c.setStrokeColor(color)
        c.setLineWidth(1.2)
        c.line(x, legend_y, x + 15, legend_y)
        marker(c, MARKERS[scenario], x + 7.5, legend_y, color, 3.0)
        c.setFillColor(axis)
        c.setFont(font, 7.3)
        c.drawString(x + 19, legend_y - 2.5, LABELS[scenario])

    # Panel a axes.
    ymin, ymax = -0.011, 0.022
    def ya(value: float) -> float:
        return left_y + (value - ymin) / (ymax - ymin) * plot_h

    def xa(percent: float) -> float:
        return left_x + (percent - 7.5) / (42.5 - 7.5) * left_w

    ticks_y = [-0.010, -0.005, 0.0, 0.005, 0.010, 0.015, 0.020]
    c.setFont(font, 7.5)
    for tick in ticks_y:
        y = ya(tick)
        c.setStrokeColor(grid)
        c.setLineWidth(0.45)
        c.line(left_x, y, left_x + left_w, y)
        c.setFillColor(axis)
        c.drawRightString(left_x - 5, y - 2.5, f"{tick:.3f}")
    c.setStrokeColor(axis)
    c.setLineWidth(0.75)
    c.line(left_x, left_y, left_x, left_y + plot_h)
    c.line(left_x, left_y, left_x + left_w, left_y)
    c.setStrokeColor(zero)
    c.setDash(3, 2.5)
    c.line(left_x, ya(0), left_x + left_w, ya(0))
    c.setDash()
    for percent in (10, 20, 40):
        x = xa(percent)
        c.setStrokeColor(axis)
        c.line(x, left_y, x, left_y - 3)
        c.setFillColor(axis)
        c.drawCentredString(x, left_y - 13, f"{percent}%")
    c.setFont(font, 8.1)
    c.drawCentredString(left_x + left_w / 2, 12, "Target initial acceptance")
    c.saveState()
    c.translate(14, left_y + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Learning-mediated exposure contrast, LMC")
    c.restoreState()

    by_scenario = {s: sorted([r for r in low if r["scenario"] == s], key=lambda r: float(r["target_initial_acceptance"])) for s in order}
    for scenario in order:
        color = hex_color(COLORS[scenario])
        points = [(xa(float(r["target_initial_acceptance"]) * 100), ya(float(r["estimate"]))) for r in by_scenario[scenario]]
        c.setStrokeColor(color)
        c.setLineWidth(1.15)
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            c.line(x0, y0, x1, y1)
        for r, (x, y) in zip(by_scenario[scenario], points):
            error_bar_vertical(c, x, y, float(r["ci_low"]), float(r["ci_high"]), ya, color)
            marker(c, MARKERS[scenario], x, y, color)

    # Panel b axes and estimates.
    xmin, xmax = -0.0055, 0.0050
    def xb(value: float) -> float:
        return right_x + (value - xmin) / (xmax - xmin) * right_w

    c.setFont(font, 7.3)
    xticks = [-0.004, -0.002, 0.0, 0.002, 0.004]
    for tick in xticks:
        x = xb(tick)
        c.setStrokeColor(grid)
        c.setLineWidth(0.45)
        c.line(x, right_y, x, right_y + plot_h)
        c.setFillColor(axis)
        c.drawCentredString(x, right_y - 13, f"{tick:.3f}")
    c.setStrokeColor(axis)
    c.setLineWidth(0.75)
    c.line(right_x, right_y, right_x + right_w, right_y)
    c.setStrokeColor(zero)
    c.setDash(3, 2.5)
    c.line(xb(0), right_y, xb(0), right_y + plot_h)
    c.setDash()
    c.setFont(font, 8.1)
    c.setFillColor(axis)
    c.drawCentredString(right_x + right_w / 2, 12, "Paired ΔLMC (refresh − fixed)")

    paired_map = {r["scenario"]: r for r in paired}
    y_positions = [155.0, 115.0, 75.0]
    for y, scenario in zip(y_positions, order):
        row = paired_map[scenario]
        color = hex_color(COLORS[scenario])
        error_bar_horizontal(c, y, float(row["ci_low"]), float(row["ci_high"]), xb, color)
        marker(c, MARKERS[scenario], xb(float(row["estimate"])), y, color)

    c.showPage()
    c.save()

    caption = (
        "Added sensitivity analyses. (a) Round-6 LMC under lower-feedback "
        "design conditions calibrated to 10%, 20%, and 40% mean initial "
        "acceptance under zero provenance response. The calibrated outside-option "
        "intercept was then held fixed across response conditions and matched "
        "branches. (b) Within-cell paired difference in round-6 LMC between "
        "roundwise candidate-pool refresh and fixed candidate membership under "
        "the original high-feedback setting. Points are estimates; bars are 95% "
        "crossed-bootstrap confidence intervals."
    )
    (OUT / "FigS_added_sensitivity_caption.txt").write_text(caption + "\n", encoding="utf-8")
    print(pdf_path)


if __name__ == "__main__":
    main()

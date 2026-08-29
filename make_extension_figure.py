#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables" / "extension"
FIGURES = ROOT / "figures" / "extension"
FIGURES.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "zero",
    "moderate_asymmetric",
    "strong_premium_penalty",
    "ai_appreciation",
]
LABELS = {
    "zero": "Zero response",
    "moderate_asymmetric": "Moderate response",
    "strong_premium_penalty": "Strong response",
    "ai_appreciation": "AI appreciation",
}
COLORS = {
    "zero": "#6B7280",
    "moderate_asymmetric": "#D97706",
    "strong_premium_penalty": "#B91C1C",
    "ai_appreciation": "#0F766E",
}


def errorbar(ax, x, row, color, marker, label=None):
    ax.errorbar(
        x,
        row.estimate,
        yerr=np.array(
            [
                [row.estimate - row.ci_low],
                [row.ci_high - row.estimate],
            ]
        ),
        color=color,
        marker=marker,
        markersize=5,
        linewidth=1.15,
        capsize=2.5,
        label=label,
    )


def main() -> None:
    lightgcn = pd.read_csv(TABLES / "lightgcn_dynamic_amplification.csv")
    trajectories = pd.read_csv(TABLES / "long_horizon_trajectories.csv")
    differences = pd.read_csv(TABLES / "long_horizon_regime_differences.csv")

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.2, 5.7))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[1, :]),
        figure.add_subplot(grid[0, 1]),
    ]

    # (a) Targeted dynamic LightGCN replication.
    ax = axes[0]
    core = [
        "moderate_asymmetric",
        "strong_premium_penalty",
        "ai_appreciation",
    ]
    x = np.arange(len(core))
    for offset, intervention, marker, label in [
        (-0.10, "none", "o", "No intervention"),
        (0.10, "combined", "s", "Combined intervention"),
    ]:
        for index, scenario in enumerate(core):
            selected = lightgcn[
                (lightgcn.scenario == scenario)
                & (lightgcn.intervention == intervention)
            ]
            if selected.empty:
                continue
            plotted_offset = (
                0.0
                if scenario == "ai_appreciation" and intervention == "none"
                else offset
            )
            errorbar(
                ax,
                index + plotted_offset,
                selected.iloc[0],
                COLORS[scenario],
                marker,
                label if index == 0 else None,
            )
    ax.axhline(0, color="#374151", linewidth=0.75, linestyle="--")
    ax.set_xticks(x, ["Moderate", "Strong", "AI appreciation"], rotation=18)
    ax.set_ylabel("Algorithmic amplification, AA")
    ax.set_title("(a) Dynamic LightGCN at round 6", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.55)

    # (b) Longest BPR trajectory under event-level online updating.
    ax = axes[1]
    selected = trajectories[
        (trajectories.horizon == 24)
        & (trajectories.update_regime == "event_online")
    ]
    for scenario in SCENARIOS:
        series = selected[selected.scenario == scenario].sort_values("round")
        ax.plot(
            series["round"],
            series["estimate"],
            color=COLORS[scenario],
            linewidth=1.4,
            label=LABELS[scenario],
        )
        ax.fill_between(
            series["round"],
            series["ci_low"],
            series["ci_high"],
            color=COLORS[scenario],
            alpha=0.12,
            linewidth=0,
        )
    ax.axhline(0, color="#374151", linewidth=0.75, linestyle="--")
    ax.set_xlabel("Simulation round")
    ax.set_title("(b) BPR-Online through round 24", loc="left")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.55)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    # (c) Paired change induced by fixed-budget history replay.
    ax = axes[2]
    for scenario in SCENARIOS[1:]:
        series = differences[differences.scenario == scenario].sort_values(
            "horizon"
        )
        ax.errorbar(
            series["horizon"],
            series["estimate"],
            yerr=np.vstack(
                [
                    series["estimate"] - series["ci_low"],
                    series["ci_high"] - series["estimate"],
                ]
            ),
            color=COLORS[scenario],
            marker="o",
            markersize=4.5,
            linewidth=1.25,
            capsize=2.5,
            label=LABELS[scenario],
        )
    ax.axhline(0, color="#374151", linewidth=0.75, linestyle="--")
    ax.set_xticks([6, 12, 24])
    ax.set_xlabel("Endpoint round")
    ax.set_ylabel("Replay − event-online AA")
    ax.set_title("(c) Fixed-budget history replay", loc="left")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.55)
    ax.legend(frameon=False, loc="best")

    figure.tight_layout(w_pad=1.15, h_pad=1.35)
    for suffix in ("png", "pdf", "svg"):
        path = FIGURES / f"SFig2_cross_architecture_horizon_regime.{suffix}"
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            save_kwargs["dpi"] = 600
        figure.savefig(path, **save_kwargs)
    plt.close(figure)


if __name__ == "__main__":
    main()

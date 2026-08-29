#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "jiis_matplotlib_cache"),
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT
CONCEPTUAL_SOURCES = ROOT / "publication_assets" / "conceptual_sources"
LOCKED_MAIN_FIGURES = ROOT / "publication_assets" / "figures" / "main"
EXTENSION = ROOT
OUT = ROOT / "publication_assets" / "generated"
SOURCE_OUT = OUT / "source_data"


COLORS = {
    "strong_premium_penalty": "#B22222",
    "moderate_asymmetric": "#D97706",
    "mild_aversion": "#0072B2",
    "heterogeneous_response": "#6B7280",
    "ai_appreciation": "#008577",
    "zero": "#6B7280",
}
LABELS = {
    "strong_premium_penalty": "Strong premium–penalty",
    "moderate_asymmetric": "Moderate asymmetry",
    "mild_aversion": "Mild aversion",
    "heterogeneous_response": "Heterogeneous response",
    "ai_appreciation": "AI appreciation",
    "zero": "Zero response",
}
MARKERS = {
    "strong_premium_penalty": "o",
    "moderate_asymmetric": "s",
    "mild_aversion": "^",
    "heterogeneous_response": "D",
    "ai_appreciation": "v",
    "zero": "D",
}
LINESTYLES = {
    "strong_premium_penalty": (0, (6, 2.2)),
    "moderate_asymmetric": "-",
    "mild_aversion": (0, (1.3, 1.8)),
    "heterogeneous_response": "-.",
    "ai_appreciation": (0, (4, 1.6, 1.2, 1.6)),
    "zero": (0, (1.3, 1.8)),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_header(ax, label: str, title: str, *, y: float = 1.035) -> None:
    ax.text(
        -0.10,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        0.06,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="normal",
        clip_on=False,
    )


def clean_axis(ax, *, grid: str = "y") -> None:
    ax.grid(axis=grid, color="#D9DDE3", linewidth=0.55)
    ax.set_axisbelow(True)


def save_figure(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg", "tif"):
        path = OUT / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in ("png", "tif"):
            kwargs["dpi"] = 600
        if suffix == "tif":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
    plt.close(fig)


def export_unchanged_figures() -> None:
    for number in (1, 2):
        source = LOCKED_MAIN_FIGURES / f"Fig{number}.pdf"
        shutil.copy2(source, OUT / source.name)

    fig1_svg = CONCEPTUAL_SOURCES / "Fig1_feedback_loop.svg"
    fig1_png = CONCEPTUAL_SOURCES / "Fig1_feedback_loop_600dpi.png"
    shutil.copy2(fig1_svg, OUT / fig1_svg.name)
    shutil.copy2(fig1_png, OUT / fig1_png.name)
    with Image.open(fig1_png).convert("RGB") as image:
        image.save(
            OUT / "Fig1_feedback_loop_600dpi.tif",
            dpi=(600, 600),
            compression="tiff_lzw",
        )

    fig2_source = CONCEPTUAL_SOURCES / "Fig2_paired_frozen_closed_design_editable_exact.pptx"
    shutil.copy2(fig2_source, OUT / fig2_source.name)


def make_fig3(roundwise: pd.DataFrame) -> None:
    order = [
        "strong_premium_penalty",
        "moderate_asymmetric",
        "mild_aversion",
        "heterogeneous_response",
        "ai_appreciation",
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.75))
    for scenario in order:
        frame = roundwise[roundwise.scenario == scenario].sort_values("round")
        ax.fill_between(
            frame["round"],
            frame["ci_low"],
            frame["ci_high"],
            color=COLORS[scenario],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            frame["round"],
            frame["estimate"],
            color=COLORS[scenario],
            linestyle=LINESTYLES[scenario],
            marker=MARKERS[scenario],
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=LABELS[scenario],
        )
    ax.axhline(0, color="#5F6670", linewidth=0.8)
    ax.set_xlim(1, 6)
    ax.set_xticks(range(1, 7))
    ax.set_ylim(-0.012, 0.040)
    ax.set_xlabel("Simulation round, t")
    ax.set_ylabel("Algorithmic amplification, AA")
    clean_axis(ax)
    panel_header(ax, "(a)", "Round-wise trajectories", y=1.02)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.00, 1.065),
        borderaxespad=0,
        columnspacing=1.35,
        handlelength=2.6,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.77)
    save_figure(fig, "Fig3_roundwise_AA_600dpi")


def make_final_round_diagnostic(final_aa: pd.DataFrame) -> None:
    order = [
        "ai_appreciation",
        "heterogeneous_response",
        "mild_aversion",
        "moderate_asymmetric",
        "strong_premium_penalty",
    ]
    frame = final_aa.set_index("scenario").loc[order].reset_index()
    y = np.arange(len(frame))[::-1]
    fig = plt.figure(figsize=(7.0, 3.55))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.66, 0.34], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    text_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    for yi, row in zip(y, frame.itertuples(index=False)):
        scenario = row.scenario
        ax.errorbar(
            row.estimate,
            yi,
            xerr=np.array(
                [[row.estimate - row.ci_low], [row.ci_high - row.estimate]]
            ),
            fmt=MARKERS[scenario],
            color=COLORS[scenario],
            markersize=5.2,
            capsize=2.5,
            linewidth=1.25,
            markeredgecolor="white",
            markeredgewidth=0.65,
        )
        text_ax.text(
            0.98,
            yi,
            f"{row.estimate:.4f} ({row.ci_low:.4f}, {row.ci_high:.4f})",
            ha="right",
            va="center",
            fontsize=8.3,
        )
    ax.axvline(0, color="#5F6670", linewidth=0.85, linestyle=(0, (3, 3)))
    ax.set_yticks(y, [LABELS[s] for s in order])
    ax.set_xlim(-0.015, 0.040)
    ax.set_xlabel("Algorithmic amplification, AA (log relative-exposure units)")
    clean_axis(ax, grid="x")
    ax.tick_params(axis="y", length=0, pad=8)
    text_ax.set_xlim(0, 1)
    text_ax.set_xticks([])
    text_ax.tick_params(axis="y", left=False, labelleft=False)
    for spine in text_ax.spines.values():
        spine.set_visible(False)
    ax.text(
        -0.42,
        1.06,
        "(b)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        -0.32,
        1.06,
        "Response scenario",
        transform=ax.transAxes,
        fontsize=9.5,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    text_ax.text(
        0.98,
        1.06,
        "AA estimate (95% CI)",
        transform=text_ax.transAxes,
        fontsize=9.5,
        ha="right",
        va="bottom",
        clip_on=False,
    )
    fig.text(
        0.52,
        0.035,
        "AI-directed  ←        →  Human-directed",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#666666",
    )
    fig.subplots_adjust(left=0.29, right=0.985, bottom=0.23, top=0.88)
    save_figure(fig, "Diagnostic_final_round_AA_600dpi")


def make_fig4(interventions: pd.DataFrame) -> None:
    scenarios = ["moderate_asymmetric", "strong_premium_penalty"]
    scenario_labels = ["Moderate", "Strong"]
    scenario_colors = ["#0072B2", "#D55E00"]
    scenario_markers = ["o", "s"]
    order = ["none", "oracle_feedback_correction", "quota_reranking", "combined"]
    labels = ["None", "Oracle correction", "Quota reranking", "Combined"]
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"wspace": 0.55})

    ax = axes[0]
    for offset, scenario, label, color, marker in zip(
        [-0.09, 0.09], scenarios, scenario_labels, scenario_colors, scenario_markers
    ):
        for yi, intervention in zip(y, order):
            row = interventions[
                (interventions.scenario == scenario)
                & (interventions.intervention == intervention)
            ].iloc[0]
            ax.errorbar(
                row.AA_estimate,
                yi + offset,
                xerr=np.array(
                    [[row.AA_estimate - row.AA_ci_low], [row.AA_ci_high - row.AA_estimate]]
                ),
                fmt=marker,
                color=color,
                markersize=5,
                capsize=2.5,
                linewidth=1.2,
                label=label if intervention == order[0] else None,
                markeredgecolor="white",
                markeredgewidth=0.55,
            )
    ax.axvline(0, color="#5F6670", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(y, labels)
    ax.set_xlabel("Algorithmic amplification, AA")
    clean_axis(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    panel_header(ax, "(a)", "Residual amplification")

    ax = axes[1]
    for offset, scenario, label, color, marker in zip(
        [-0.09, 0.09], scenarios, scenario_labels, scenario_colors, scenario_markers
    ):
        for yi, intervention in zip(y, order):
            if intervention == "none":
                estimate = low = high = 0.0
            else:
                row = interventions[
                    (interventions.scenario == scenario)
                    & (interventions.intervention == intervention)
                ].iloc[0]
                estimate = row.delta_candidate_anchored_utility_estimate
                low = row.delta_candidate_anchored_utility_ci_low
                high = row.delta_candidate_anchored_utility_ci_high
            xerr = None if intervention == "none" else np.array([[estimate - low], [high - estimate]])
            ax.errorbar(
                estimate,
                yi + offset,
                xerr=xerr,
                fmt=marker,
                color=color,
                markersize=5,
                capsize=2.5,
                linewidth=1.2,
                markeredgecolor="white",
                markeredgewidth=0.55,
            )
    ax.axvline(0, color="#5F6670", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(y, labels)
    ax.set_xlabel("Δ candidate-anchored utility (SD units)")
    clean_axis(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    panel_header(ax, "(b)", "Utility contrast vs. no intervention")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.01),
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.text(
        0.76,
        0.03,
        "Reference row fixed at zero by construction",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#666666",
    )
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.17, top=0.83)
    save_figure(fig, "Fig4_intervention_AA_utility_600dpi")


def sensitivity_panel(ax, frame: pd.DataFrame, x_col: str, x_values: list[float]) -> None:
    order = ["ai_appreciation", "moderate_asymmetric", "strong_premium_penalty"]
    for scenario in order:
        series = frame[frame.scenario == scenario].sort_values(x_col)
        ax.errorbar(
            series[x_col],
            series["estimate"],
            yerr=np.vstack(
                [
                    series["estimate"] - series["ci_low"],
                    series["ci_high"] - series["estimate"],
                ]
            ),
            color=COLORS[scenario],
            marker=MARKERS[scenario],
            linestyle=LINESTYLES[scenario],
            markersize=4.5,
            capsize=2.5,
            linewidth=1.25,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=LABELS[scenario].replace(" asymmetry", ""),
        )
    ax.axhline(0, color="#5F6670", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xticks(x_values)
    ax.set_ylim(-0.06, 0.115)
    clean_axis(ax)


def make_sfig1(candidate: pd.DataFrame, updates: pd.DataFrame, oracle: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.55), sharey=True, gridspec_kw={"wspace": 0.34})
    sensitivity_panel(axes[0], candidate, "candidate_size", [240, 500, 1000])
    sensitivity_panel(axes[1], updates, "online_steps", [1, 3, 10])
    sensitivity_panel(axes[2], oracle, "oracle_c", [0.5, 1.0, 1.5])
    panel_header(axes[0], "(a)", "Candidate pool, M")
    panel_header(axes[1], "(b)", "Online updates, K")
    panel_header(axes[2], "(c)", "Oracle multiplier, c")
    axes[0].set_ylabel("Algorithmic amplification, AA")
    axes[1].set_xlabel("Sensitivity setting")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.01),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.80)
    save_figure(fig, "SFig1_sensitivity_600dpi")


def reconcile_common_baseline(
    candidate: pd.DataFrame,
    updates: pd.DataFrame,
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use one fixed-seed CI for the common M=240, K=3 event-online units."""
    reference = canonical[
        (canonical["core"] == "confirmatory_10u5i")
        & (canonical["update_regime"] == "event_online")
    ].set_index("scenario")
    candidate = candidate.copy()
    updates = updates.copy()
    for scenario, row in reference.iterrows():
        candidate_mask = (
            (candidate["candidate_size"] == 240)
            & (candidate["scenario"] == scenario)
        )
        update_mask = (
            (updates["online_steps"] == 3)
            & (updates["scenario"] == scenario)
        )
        for frame, mask in ((candidate, candidate_mask), (updates, update_mask)):
            frame.loc[mask, "estimate"] = row["estimate"]
            frame.loc[mask, "ci_low"] = row["ci_low"]
            frame.loc[mask, "ci_high"] = row["ci_high"]
    return candidate, updates


def plot_lightgcn(ax, lightgcn: pd.DataFrame) -> None:
    core = ["moderate_asymmetric", "strong_premium_penalty", "ai_appreciation"]
    x = np.arange(len(core))
    for offset, intervention, marker, intervention_label in [
        (-0.10, "none", "o", "No intervention"),
        (0.10, "combined", "s", "Combined intervention"),
    ]:
        for index, scenario in enumerate(core):
            selected = lightgcn[
                (lightgcn.scenario == scenario) & (lightgcn.intervention == intervention)
            ]
            if selected.empty:
                continue
            row = selected.iloc[0]
            plotted_offset = 0.0 if scenario == "ai_appreciation" else offset
            ax.errorbar(
                index + plotted_offset,
                row.estimate,
                yerr=np.array([[row.estimate - row.ci_low], [row.ci_high - row.estimate]]),
                color=COLORS[scenario],
                marker=marker,
                linestyle="none",
                markersize=5,
                capsize=2.5,
                linewidth=1.2,
                label=intervention_label if index == 0 else None,
                markeredgecolor="white",
                markeredgewidth=0.55,
            )
    ax.axhline(0, color="#5F6670", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xticks(x, ["Moderate", "Strong", "AI appreciation"], rotation=15)
    ax.set_ylabel("Algorithmic amplification, AA")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4)


def plot_replay(ax, differences: pd.DataFrame) -> None:
    for scenario in ["moderate_asymmetric", "strong_premium_penalty", "ai_appreciation"]:
        series = differences[differences.scenario == scenario].sort_values("horizon")
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
            marker=MARKERS[scenario],
            linestyle=LINESTYLES[scenario],
            markersize=4.5,
            capsize=2.5,
            linewidth=1.25,
            label=LABELS[scenario].replace(" asymmetry", " response").replace(" premium–penalty", " response"),
            markeredgecolor="white",
            markeredgewidth=0.55,
        )
    ax.axhline(0, color="#5F6670", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xticks([6, 12, 24])
    ax.set_xlabel("Endpoint round")
    ax.set_ylabel("Paired ΔAA (history replay − event-online)")
    clean_axis(ax)
    ax.legend(frameon=False, loc="best", handletextpad=0.4)


def plot_horizon(ax, trajectories: pd.DataFrame) -> None:
    selected = trajectories[
        (trajectories.horizon == 24) & (trajectories.update_regime == "event_online")
    ]
    for scenario in ["zero", "moderate_asymmetric", "strong_premium_penalty", "ai_appreciation"]:
        series = selected[selected.scenario == scenario].sort_values("round")
        ax.plot(
            series["round"],
            series["estimate"],
            color=COLORS[scenario],
            linestyle=LINESTYLES[scenario],
            marker=MARKERS[scenario],
            markevery=[0, 5, 11, 17, 23],
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=LABELS[scenario].replace(" asymmetry", " response").replace(" premium–penalty", " response"),
        )
        ax.fill_between(
            series["round"],
            series["ci_low"],
            series["ci_high"],
            color=COLORS[scenario],
            alpha=0.11,
            linewidth=0,
        )
    ax.axhline(0, color="#5F6670", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xlabel("Simulation round")
    ax.set_ylabel("Algorithmic amplification, AA")
    clean_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=2.6, handletextpad=0.4)


def make_sfig2(lightgcn: pd.DataFrame, trajectories: pd.DataFrame, differences: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 5.7))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], wspace=0.42, hspace=0.48)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    plot_lightgcn(ax_a, lightgcn)
    plot_replay(ax_b, differences)
    plot_horizon(ax_c, trajectories)
    panel_header(ax_a, "(a)", "Dynamic LightGCN at round 6")
    panel_header(ax_b, "(b)", "Fixed-budget history replay")
    panel_header(ax_c, "(c)", "BPR-Online through round 24", y=1.03)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.95)
    save_figure(fig, "SFig2_targeted_validation_600dpi")

    for label, plotter, width in [
        ("SFig2a_Dynamic_LightGCN_round6_600dpi", lambda ax: plot_lightgcn(ax, lightgcn), 5.8),
        ("SFig2b_history_replay_600dpi", lambda ax: plot_replay(ax, differences), 5.8),
        ("SFig2c_BPR_Online_round24_600dpi", lambda ax: plot_horizon(ax, trajectories), 7.2),
    ]:
        fig_single, ax_single = plt.subplots(figsize=(width, 4.0 if width < 7 else 4.25))
        plotter(ax_single)
        panel = {"SFig2a": ("(a)", "Dynamic LightGCN at round 6"), "SFig2b": ("(b)", "Fixed-budget history replay"), "SFig2c": ("(c)", "BPR-Online through round 24")}
        key = label[:6]
        panel_header(ax_single, panel[key][0], panel[key][1])
        fig_single.subplots_adjust(left=0.14 if width < 7 else 0.11, right=0.985, bottom=0.16, top=0.88)
        save_figure(fig_single, label)


def copy_sources(paths: list[Path]) -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, SOURCE_OUT / path.name)
    shutil.copy2(Path(__file__), SOURCE_OUT / Path(__file__).name)


def write_readme() -> None:
    (OUT / "README.txt").write_text(
        "JIIS figure-regeneration set. The exact submitted Figs. 1–4 are retained in "
        "publication_assets/figures/main. Fig. 3, Fig. 4, SFig. 1, and SFig. 2 can also be "
        "regenerated from the included CSV files using a common Arial visual system. "
        "The common M=240, K=3 event-online cells in SFig. 1 use the single fixed-seed "
        "canonical intervals reported in robustness_endpoints.csv. "
        "Raster files are 600 dpi; PDF and SVG files are vector exports.\n\n"
        "Online Resource 1 SFig. 2 caption:\n"
        "SFig. 2. Targeted validation across architecture, horizon, and update regime. "
        "(a) Final-round LightGCN amplification under no intervention and the combined "
        "intervention. (b) Paired change in amplification when one of three per-event "
        "updates was allocated to historical replay. (c) BPR-Online trajectories "
        "through 24 rounds under event-level updating. Points and lines denote "
        "estimates; error bars and bands denote 95% crossed-bootstrap confidence "
        "intervals from the reduced validation grid.\n",
        encoding="utf-8",
    )


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)

    roundwise_path = V1 / "tables/Fig3_roundwise_AA_data.csv"
    final_aa_path = V1 / "tables/Table4_BPR_algorithmic_amplification.csv"
    interventions_path = V1 / "tables/Table5_intervention_tradeoffs.csv"
    candidate_path = V1 / "tables/candidate_pool_sensitivity.csv"
    updates_path = V1 / "tables/online_update_sensitivity.csv"
    oracle_path = V1 / "tables/oracle_misspecification_sensitivity.csv"
    lightgcn_path = EXTENSION / "tables/extension/lightgcn_dynamic_amplification.csv"
    trajectories_path = EXTENSION / "tables/extension/long_horizon_trajectories.csv"
    differences_path = EXTENSION / "tables/extension/long_horizon_regime_differences.csv"
    canonical_path = ROOT / "tables/robustness/robustness_endpoints.csv"

    candidate = pd.read_csv(candidate_path)
    updates = pd.read_csv(updates_path)
    canonical = pd.read_csv(canonical_path)
    candidate, updates = reconcile_common_baseline(candidate, updates, canonical)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(SOURCE_OUT / "candidate_pool_sensitivity_reconciled.csv", index=False)
    updates.to_csv(SOURCE_OUT / "online_update_sensitivity_reconciled.csv", index=False)

    export_unchanged_figures()
    make_fig3(pd.read_csv(roundwise_path))
    make_final_round_diagnostic(pd.read_csv(final_aa_path))
    make_fig4(pd.read_csv(interventions_path))
    make_sfig1(
        candidate,
        updates,
        pd.read_csv(oracle_path),
    )
    make_sfig2(
        pd.read_csv(lightgcn_path),
        pd.read_csv(trajectories_path),
        pd.read_csv(differences_path),
    )
    copy_sources(
        [
            roundwise_path,
            final_aa_path,
            interventions_path,
            candidate_path,
            updates_path,
            oracle_path,
            lightgcn_path,
            trajectories_path,
            differences_path,
            canonical_path,
        ]
    )
    write_readme()
    print(OUT)


if __name__ == "__main__":
    main()

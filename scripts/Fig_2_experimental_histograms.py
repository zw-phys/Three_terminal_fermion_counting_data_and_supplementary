import csv
import shutil
from pathlib import Path
from runpy import run_path

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib import rc_context, rcParams
from matplotlib.figure import Figure
from matplotlib.patches import Patch


SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
PREISS_HEATMAP = DATA_DIR / "preiss_fig2a_native_raster.png"
PREISS_COLORBAR = DATA_DIR / "preiss_fig2a_colorbar.png"
PREISS_SCRIPT = SCRIPTS_DIR / "Fig_2_and_SupFig_2_Preiss_digitization.py"
BOJOVIC_SCRIPT = SCRIPTS_DIR / "Fig_2_and_SupFig_3_Bojovic_frequencies.py"
PHASE_OFFSET = 0.0
AXIS_LIMIT = 3.7
USE_TEX = False

FIGSIZE = (3.375, 2.95)
DPI = 600
FONT_SIZE = 10.0
MARKER_SIZE = 4.2
LINEWIDTH = 0.8
colors = ["#0D2DFF", "#F20D0D", "#8C8C8C"]
markers = ["o", "s", "^"]
datasets = ("Preiss", "Bojović (30)", "Bojović (50)")
tex_labels = ("Preiss", r"Bojovi\'{c} (30)", r"Bojovi\'{c} (50)")
outcomes = ["200", "020", "002", "110", "101", "011"]


def calculate_margins(histograms):
    if histograms.shape != (3, 6):
        raise ValueError("需要三组完整的六项概率.")
    if not np.isfinite(histograms).all() or (histograms < 0).any():
        raise ValueError("概率必须有限且非负.")
    if not np.allclose(histograms.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("各组概率必须已经归一化; 程序不重新归一化.")

    p1, p2, p3, s12, s13, s23 = histograms.T
    pairwise = np.column_stack((s12**2-4*p1*p2,
    s13**2-4*p1*p3,
    s23**2-4*p2*p3))
    Delta3 = (4*p1*p2*p3 + s12*s13*s23
    - p1*s23**2 - p2*s13**2 - p3*s12**2)
    G = np.array([[p1, s12/2, s13/2],
    [s12/2, p2, s23/2],
    [s13/2, s23/2, p3]]).transpose(2, 0, 1)
    if not np.allclose(Delta3, 4*np.linalg.det(G), rtol=1e-12, atol=1e-14):
        raise ArithmeticError("三终端裕量与矩阵行列式不一致.")
    return pairwise, Delta3


def configure_style(usetex):
    if usetex:
        missing = [name for name in ("latex", "dvipng") if shutil.which(name) is None]
        if missing:
            raise RuntimeError("TeX commands required: " + ", ".join(missing))
    rcParams.update({
    "text.usetex": usetex,
    "font.family": "serif",
    "font.serif": ["Times"] if usetex else ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FONT_SIZE,
    "axes.labelsize": 11,
    "axes.labelpad": 3,
    "axes.linewidth": LINEWIDTH,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.6,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "legend.fontsize": FONT_SIZE,
    "legend.frameon": False,
    "lines.markersize": MARKER_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": DPI,
    "savefig.facecolor": "white",
    })


def draw_histograms(ax, histograms, frequency_errors, usetex):
    x = np.arange(6)
    for i, row in enumerate(histograms):
        positions = x + (i-1)*0.25
        ax.bar(positions, row, width=0.21, color=colors[i],
        edgecolor=colors[i], linewidth=0.6, zorder=2)
        if i > 0:
            errors = ax.errorbar(positions[2:], row[2:], yerr=frequency_errors[i-1],
            fmt="none", ecolor="black", elinewidth=0.6,
            capsize=1.3, capthick=0.6, zorder=3)
            for artist in (*errors[1], *errors[2]):
                artist.set_path_effects([
                path_effects.withStroke(linewidth=1.2, foreground="white")
                ])
    ax.set(xlim=(-0.55, 5.55), ylim=(-0.035, 0.70),
    xticks=x, xticklabels=outcomes,
    yticks=[0, 0.2, 0.4, 0.6],
    ylabel="Probability", xlabel="Two-particle outcome")
    ax.text(0.03, 0.95, r"\textbf{(a)}" if usetex else "(a)", transform=ax.transAxes,
    ha="left", va="top", fontsize=11, fontweight="bold")


def draw_margin(ax, Delta3, legend_labels, usetex):
    ax.axhline(0, color="0.45", linewidth=0.8, linestyle="--", zorder=1)
    for i, value in enumerate(Delta3):
        ax.plot(i, value, color=colors[i], marker=markers[i], linestyle="none",
        markersize=MARKER_SIZE, markeredgewidth=0.6, zorder=3)
    ax.set(xlim=(-0.50, 2.50), ylim=(-0.155, 0.055),
    xticks=[0, 1, 2],
    xticklabels=legend_labels,
    yticks=[-0.15, -0.10, -0.05, 0, 0.05])
    ax.set_ylabel(r"$\Delta_3$", fontsize=14, labelpad=4)
    ax.text(0.03, 0.95, r"\textbf{(b)}" if usetex else "(b)", transform=ax.transAxes,
    ha="left", va="top", fontsize=11, fontweight="bold")


def generate(figures_dir, csv_dir, preiss_result, bojovic_results, usetex=False):
    preiss = preiss_result["primary_result"]
    bojovic_probabilities, frequency_errors = bojovic_results
    histograms = np.vstack((preiss["p"] + preiss["s"], bojovic_probabilities))
    pairwise, Delta3 = calculate_margins(histograms)
    if (pairwise < -1e-12).any():
        raise ValueError("本图数据须满足三个成对相容条件.")
    export_data(csv_dir, histograms, frequency_errors, pairwise, Delta3)
    with rc_context():
        configure_style(usetex)
        render(figures_dir, histograms, frequency_errors, Delta3, usetex)


def render(figures_dir, histograms, frequency_errors, Delta3, usetex):
    legend_labels = tex_labels if usetex else datasets
    fig = Figure(figsize=FIGSIZE, dpi=DPI)
    ax_a = fig.add_axes((0.215, 0.605, 0.765, 0.375))
    ax_b = fig.add_axes((0.215, 0.100, 0.765, 0.325))
    draw_histograms(ax_a, histograms, frequency_errors, usetex)
    draw_margin(ax_b, Delta3, legend_labels, usetex)
    for ax in (ax_a, ax_b):
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", which="both", right=False)

    handles = [Patch(facecolor=color, edgecolor=color, linewidth=0.6, label=label)
    for color, label in zip(colors, legend_labels)]
    ax_a.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.79),
    borderaxespad=0, borderpad=0, handlelength=0.7,
    handletextpad=0.35, labelspacing=0.2)

    fig.savefig(figures_dir / "Fig_2.png", dpi=DPI)


def export_data(csv_dir, histograms, frequency_errors, pairwise, delta3):
    with (csv_dir / "fig2_probabilities.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("dataset", "outcome", "probability", "error_68_lower", "error_68_upper"))
        for i, row in enumerate(histograms):
            for j, probability in enumerate(row):
                errors = frequency_errors[i-1, :, j-2] if i > 0 and j >= 2 else ("", "")
                writer.writerow((datasets[i], outcomes[j], probability, *errors))

    with (csv_dir / "fig2_margins.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("dataset", "delta12", "delta13", "delta23", "delta3"))
        for dataset, margins, value in zip(datasets, pairwise, delta3):
            writer.writerow((dataset, *margins, value))


if __name__ == "__main__":
    preiss_functions = run_path(str(PREISS_SCRIPT))
    bojovic_functions = run_path(str(BOJOVIC_SCRIPT))
    intensity, _ = preiss_functions["invert_colorbar"](PREISS_HEATMAP, PREISS_COLORBAR)
    preiss_result = preiss_functions["terminal_statistics"](
    intensity, phase_offset=PHASE_OFFSET, axis_limit=AXIS_LIMIT,
    )
    bojovic_results = bojovic_functions["run"]()
    generate(
    SCRIPTS_DIR, SCRIPTS_DIR, {"primary_result": preiss_result}, bojovic_results,
    usetex=USE_TEX,
    )

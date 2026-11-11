import csv
from pathlib import Path
from runpy import run_path

from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
OUTPUT_DIR = SCRIPTS_DIR
PREISS_HEATMAP = DATA_DIR / "preiss_fig2a_native_raster.png"
PREISS_COLORBAR = DATA_DIR / "preiss_fig2a_colorbar.png"
PREISS_SCRIPT = SCRIPTS_DIR / "Fig_2_and_SupFig_2_Preiss_digitization.py"
BOJOVIC_SCRIPT = SCRIPTS_DIR / "Fig_2_and_SupFig_3_Bojovic_frequencies.py"
PHASE_OFFSET = 0.0
AXIS_LIMIT = 3.7
DPI = 250
BLUE = "#2764a0"
RED = "#b0493e"
GRAY = "#6e757b"
GREEN = "#30836f"
GOLD = "#dfb357"



def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


preiss_functions = run_path(str(PREISS_SCRIPT))
bojovic_functions = run_path(str(BOJOVIC_SCRIPT))
intensity, _ = preiss_functions["invert_colorbar"](PREISS_HEATMAP, PREISS_COLORBAR)
preiss = preiss_functions["terminal_statistics"](
intensity, phase_offset=PHASE_OFFSET, axis_limit=AXIS_LIMIT,
)
bojovic_results = bojovic_functions["run"]()

bojovic_probabilities, errors = bojovic_results
histograms = np.vstack((preiss["p"] + preiss["s"], bojovic_probabilities))
outcomes = ("200", "020", "002", "110", "101", "011")
labels = ("Preiss", "Bojović (30)", "Bojović (50)")
separation = histograms[1:, 4:].sum(axis=1)
bound = 2 * np.sqrt(histograms[1:, 2] * histograms[1:, 3])
delta3 = preiss["C"]

rows = []
for dataset, (label, probabilities) in enumerate(zip(labels, histograms)):
    for index, (outcome, probability) in enumerate(zip(outcomes, probabilities)):
        limits = errors[dataset - 1, :, index - 2] if dataset and index >= 2 else ("", "")
        rows.append((label, outcome, probability, *limits))
write_csv(OUTPUT_DIR / "SupFig3_histograms.csv",
("dataset", "outcome", "probability", "error_68pct_lower", "error_68pct_upper"),
rows)
write_csv(OUTPUT_DIR / "SupFig3_preiss_commutator.csv",
("delta3", "commutator_frobenius_norm_squared"), [(delta3, 2 * delta3)])
write_csv(OUTPUT_DIR / "SupFig3_separation.csv",
("pulse_duration_us", "separated_outcomes", "one_body_lower_bound"),
zip((30, 50), separation, bound))
write_csv(OUTPUT_DIR / "SupFig3_sectors.csv",
("terminal", "momentum_sector_start", "momentum_sector_end", "spin_well_modes"),
[(1, 0, 2/3, "L up"), (2, 2/3, 4/3, "L down"), (3, 4/3, 2, "R up; R down")])

with rc_context({
"font.family": "DejaVu Serif",
"font.size": 7.2,
"mathtext.fontset": "dejavuserif",
"axes.labelsize": 8.2,
"axes.titlesize": 8.2,
"axes.linewidth": 0.65,
"xtick.direction": "in",
"ytick.direction": "in",
"xtick.major.size": 3.4,
"ytick.major.size": 3.4,
"xtick.major.width": 0.65,
"ytick.major.width": 0.65,
"legend.frameon": False,
"legend.handlelength": 1.2,
"legend.labelspacing": 0.25,
"hatch.linewidth": 0.9,
"text.usetex": False,
}):
    figure = Figure(figsize=(843 / DPI, 1000 / DPI))
    sectors = figure.add_axes([0.15, 0.78, 0.82, 0.20])
    sectors.set_axis_off()
    sectors.text(0.23, 0.93, "Momentum sectors", ha="center")
    for index, color in enumerate((BLUE, GOLD, GREEN)):
        left = 0.04 + index * 0.13
        sectors.add_patch(Rectangle((left, 0.61), 0.13, 0.16, color=color, lw=0))
        sectors.text(left + 0.065, 0.69, str(index + 1), color="white",
        ha="center", va="center")
    sectors.annotate("", xy=(0.45, 0.55), xytext=(0.025, 0.55),
    arrowprops={"arrowstyle": "->", "lw": 0.65})
    sectors.text(0.04, 0.44, "0", ha="center")
    sectors.text(0.42, 0.44, "2", ha="center")
    sectors.text(0.23, 0.31, r"$q=k/k_{\mathrm{lat}}$ (period 2)", ha="center")
    sectors.text(0.715, 0.93, "Spin / well sectors", ha="center")
    sectors.text(0.58, 0.72, r"$L\!\uparrow\ \rightarrow\ 1$")
    sectors.text(0.58, 0.52, r"$L\!\downarrow\ \rightarrow\ 2$")
    sectors.text(0.58, 0.32, r"$\{R\!\uparrow, R\!\downarrow\}\ \rightarrow\ 3$")
    for letter, y in zip("abc", (0.985, 0.75, 0.345)):
        figure.text(0.012, y, f"({letter})", ha="left", va="top", fontsize=8, fontweight="bold")

    ax = figure.add_axes([0.15, 0.437, 0.82, 0.283])
    positions = np.arange(len(outcomes))
    for index, (color, hatch, row) in enumerate(
    zip((BLUE, RED, GRAY), ("", "//", ".."), histograms)
    ):
        x = positions + (index - 1) * 0.25
        ax.bar(x, row, width=0.24, color=color, hatch=hatch,
        edgecolor="white", linewidth=0.45)
        if index:
            ax.errorbar(x[2:], row[2:], yerr=errors[index - 1], fmt="none",
            ecolor="black", elinewidth=0.6, capsize=1.2,
            capthick=0.8, zorder=3)
    handles = [
    Patch(facecolor=color, edgecolor="white", hatch=hatch, label=label)
    for color, hatch, label in zip((BLUE, RED, GRAY), ("", "//", ".."), labels)
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.1, borderaxespad=0.35)
    ax.text(0.98, 0.965, rf"Preiss: $\Delta_3={delta3:.4f}$" "\n"
    rf"$\Vert[A_i,A_j]\Vert_{{\mathrm{{F}}}}^2={2 * delta3:.4f}$",
    transform=ax.transAxes, ha="right", va="top", color=BLUE, fontsize=6.4)
    ax.set(xlim=(-0.65, 5.65), ylim=(0, 0.90), xticks=positions, xticklabels=outcomes,
    yticks=(0, 0.2, 0.4, 0.6), ylabel="Probability", xlabel="Two-particle outcome")

    ax = figure.add_axes([0.15, 0.10, 0.82, 0.217])
    x = np.array([30, 50])
    ax.vlines(x, separation, bound, color="#bac0c6", lw=1)
    ax.scatter(x, separation, s=22, c=RED, zorder=3, label="Separated outcomes")
    ax.hlines(bound, x - 1.3, x + 1.3, color=BLUE, lw=1.5, label="One-body bound")
    ax.legend(loc="center", fontsize=6.5)
    ax.set(xlim=(19, 61), ylim=(0, 1.02), xticks=x, yticks=(0, 0.5, 1.0),
    ylabel="Probability", xlabel=r"Pulse duration ($\mu$s)")
    figure.savefig(OUTPUT_DIR / "SupFig3.png", dpi=DPI, facecolor="white")

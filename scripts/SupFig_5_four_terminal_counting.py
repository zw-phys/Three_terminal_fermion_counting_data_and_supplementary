import csv
from collections import defaultdict
from itertools import product
from math import sqrt
from pathlib import Path

from matplotlib import rc_context
from matplotlib.figure import Figure
import numpy as np


BLUE = "#2764a0"
RED = "#b0493e"
GREEN = "#30836f"
DPI = 250
FIGURE_SIZE = (843 / DPI, 675 / DPI)
OUTPUT_DIR = Path(__file__).resolve().parent


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def independent_distribution(channels):
    terminals, particles = channels.shape
    if np.min(channels) < 0 or not np.allclose(channels.sum(axis=0), 1):
        raise ValueError("Each channel must define a normalized probability distribution.")
    distribution = defaultdict(float)
    for destinations in product(range(terminals), repeat=particles):
        occupation = tuple(destinations.count(i) for i in range(terminals))
        distribution[occupation] += np.prod(channels[destinations, range(particles)])
    return dict(distribution)


def grouped_distributions(csv_dir, name, target, channels, groups, group_for, labels):
    reference = independent_distribution(channels)
    occupations = sorted(reference, reverse=True)
    counts = np.array(occupations)
    target_values = np.array([target.get(occupation, 0.0) for occupation in occupations])
    reference_values = np.array([reference[occupation] for occupation in occupations])
    if not np.allclose([target_values.sum(), reference_values.sum()], 1):
        raise ValueError(f"{name}: a full counting distribution must sum to one.")

    delta = target_values - reference_values
    if not (
    np.allclose(counts.T @ delta, 0, atol=1e-12)
    and np.allclose(counts.T @ (counts * delta[:, None]), 0, atol=1e-12)
    ):
        raise ArithmeticError(f"{name}: the model does not match the stated moments.")

    occupation_groups = [group_for(occupation) for occupation in occupations]
    write_csv(csv_dir / f"{name}_full_probabilities.csv",
    [*(f"n{i + 1}" for i in range(counts.shape[1])), "event_group", *labels],
    [(*occupation, group, first, second) for occupation, group, first, second in
    zip(occupations, occupation_groups, target_values, reference_values)])
    grouped = [
    [values[np.array(occupation_groups) == group].sum() for group in groups]
    for values in (target_values, reference_values)
    ]
    write_csv(csv_dir / f"{name}_grouped_probabilities.csv",
    ("event_group", *labels), zip(groups, *grouped))
    return grouped


def paired_bars(ax, groups, values, legend):
    x = np.arange(len(groups))
    ax.bar(x - 0.16, values[0], 0.31, label=legend[0], color=RED,
    linewidth=0.5, edgecolor="white")
    ax.bar(x + 0.16, values[1], 0.31, label=legend[1], color=BLUE, hatch="//",
    linewidth=0.5, edgecolor="white")
    ax.set(xticks=x, xticklabels=groups, ylabel="Probability")


def panel_label(figure, y, letter):
    figure.text(0.012, y, f"({letter})", ha="left", va="top",
    fontsize=8, fontweight="bold")


def vamos_event_group(occupation):
    if occupation == (2, 2, 0, 0):
        return "2200"
    if tuple(sorted(occupation, reverse=True)) == (2, 1, 1, 0):
        return "perm. 2110"
    if occupation == (1, 1, 1, 1):
        return "1111"
    return "remaining"



alpha = (16 - sqrt(118)) / 65
beta = alpha + 1/130
gamma = sqrt(34/65)
eta = sqrt(32/65)
left_plus = (1 - 2 * alpha + gamma) / 2
left_minus = (1 - 2 * alpha - gamma) / 2
right_plus = (1 - 2 * beta + eta) / 2
right_minus = (1 - 2 * beta - eta) / 2
channels = np.array([
[left_plus, left_minus, beta, beta],
[left_minus, left_plus, beta, beta],
[alpha, alpha, right_plus, right_minus],
[alpha, alpha, right_minus, right_plus],
])
target = {(2, 2, 0, 0): 1/65, (1, 1, 1, 1): 16/65}
for doubled in range(4):
    for empty in range(4):
        if doubled != empty:
            occupation = [1] * 4
            occupation[doubled] = 2
            occupation[empty] = 0
            target[tuple(occupation)] = 4/65
groups = ("2200", "perm. 2110", "1111", "remaining")
values = grouped_distributions(
OUTPUT_DIR, "SupFig5", target, channels, groups, vamos_event_group,
("chi_V", "chi_star"),
)

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
    figure = Figure(figsize=FIGURE_SIZE)
    panel_label(figure, 0.99, "a")
    panel_label(figure, 0.57, "b")
    top = figure.add_axes([0.15, 0.68, 0.82, 0.30])
    top.set_axis_off()
    top.text(0.5, 0.98, r"Four particles, four terminals: $\chi_V$", fontsize=8,
    ha="center", va="top")
    for x, color, label in (
    (0.13, GREEN, "Each three-terminal\nmerging\nrealizable"),
    (0.50, BLUE, "First + second\nmoments\nmatched by " r"$\chi_*$"),
    (0.86, RED, "No four-terminal\nrealization\n" r"$15<16$"),
    ):
        top.annotate("", xy=(x, 0.52), xytext=(0.50, 0.73),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8})
        top.text(x, 0.43, label, ha="center", va="top", color=color, fontsize=7.1)

    ax = figure.add_axes([0.15, 0.13, 0.82, 0.40])
    paired_bars(ax, groups, values, (r"$\chi_V$", r"$\chi_*$"))
    ax.set(ylim=(0, 0.90), yticks=(0, 0.3, 0.6))
    ax.text(0.03, 0.96, "Same means and\nsecond moments", ha="left", va="top",
    transform=ax.transAxes, fontsize=6.8)
    ax.legend(loc="upper right", fontsize=8)
    figure.savefig(OUTPUT_DIR / "SupFig5.png", dpi=DPI, facecolor="white")

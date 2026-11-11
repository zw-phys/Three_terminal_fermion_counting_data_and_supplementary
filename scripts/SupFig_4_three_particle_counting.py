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
DPI = 250
FIGURE_SIZE = (843 / DPI, 512 / DPI)
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



u = (2 - sqrt(2)) / 6
v = (1 + sqrt(2)) / 3
channels = np.full((3, 3), u)
np.fill_diagonal(channels, v)
target = {(3, 0, 0): 1/18, (0, 3, 0): 1/18, (0, 0, 3): 1/18, (1, 1, 1): 5/6}
groups = ("perm. 300", "perm. 210", "111")
group_by_maximum = {3: groups[0], 2: groups[1], 1: groups[2]}
values = grouped_distributions(
OUTPUT_DIR, "SupFig4", target, channels, groups,
lambda occupation: group_by_maximum[max(occupation)], ("chi_3", "chi_3_star"),
)
expected = [(1/6, 0, 5/6),
((sqrt(2) - 1) / 18, (4 - sqrt(2)) / 6, (7 + 2 * sqrt(2)) / 18)]
if not np.allclose(values, expected, rtol=0, atol=1e-12):
    raise ArithmeticError("SupFig4: grouped probabilities differ from the caption.")

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
    ax = figure.add_axes([0.16, 0.18, 0.81, 0.73])
    paired_bars(ax, groups, values, (r"$\chi_3$", r"$\chi_{3,*}$"))
    ax.set(ylim=(0, 1), yticks=np.linspace(0, 1, 5))
    ax.set_title("Identical first and second moments", pad=3.0)
    ax.legend(loc="upper left", fontsize=8)
    figure.savefig(OUTPUT_DIR / "SupFig4.png", dpi=DPI, facecolor="white")

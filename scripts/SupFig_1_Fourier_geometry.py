import csv
from pathlib import Path

from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

DPI = 250
OUTPUT_DIR = Path(__file__).resolve().parent
SOURCE_IMAGE = OUTPUT_DIR.parent / "data" / "supfig1_original.png"
FIGURE_PATH = OUTPUT_DIR / "SupFig1.png"

with Image.open(SOURCE_IMAGE) as source_image:
    original = np.asarray(source_image.convert("RGB"))
height, width = original.shape[:2]
with rc_context({
"font.family": "serif",
"font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
"font.weight": "normal",
"mathtext.fontset": "stix",
"text.usetex": False,
}):
    fig = Figure(figsize=((width + 1e-6) / DPI, (height + 1e-6) / DPI),
    dpi=DPI)
    fig.figimage(original, xo=0, yo=0, origin="upper", zorder=0)
    boxes = ((824, 160, 879, 217), (720, 285, 990, 329))
    colors = (original[190, 825] / 255, "white")
    for (x0, y0, x1, y1), color in zip(boxes, colors):
        fig.add_artist(Rectangle(
        (x0 / width, 1 - y1 / height),
        (x1 - x0) / width, (y1 - y0) / height,
        transform=fig.transFigure, facecolor=color, edgecolor="none",
        linewidth=0, antialiased=False, zorder=1,
        ))
    for x, y, label, size in (
    (851, 190, r"$\mathcal{A}$", 12),
    (851, 309, r"$\Delta_3=16\mathcal{A}^2=1/27$", 9.5),
    ):
        fig.text(x / width, 1 - y / height, label, fontsize=size,
        ha="center", va="center", color="black", zorder=2,
        math_fontfamily="cm")
    fig.savefig(FIGURE_PATH, dpi=DPI)

with Image.open(FIGURE_PATH) as output_image:
    updated = np.asarray(output_image.convert("RGB"))
if updated.shape != original.shape:
    raise RuntimeError("The original canvas dimensions must be preserved.")
outside = np.ones((height, width), dtype=bool)
for x0, y0, x1, y1 in boxes:
    outside[y0 - 2:y1 + 2, x0 - 2:x1 + 2] = False
if not np.array_equal(original[outside], updated[outside]):
    raise RuntimeError("Pixels outside the two area-label regions changed.")

outcomes = ("200", "020", "002", "110", "101", "011")
with (OUTPUT_DIR / "supfig1_fourier.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(("outcome", "probability"))
    writer.writerows(zip(outcomes, (0, 0, 0, 1/3, 1/3, 1/3)))

labeled_points = {(0, 0): "F", (80, 0): "C", (80, 20): "X"}
with (OUTPUT_DIR / "supfig1_fixed_means.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(("p1", "p2", "p3", "s12", "s13", "s23", "delta23", "delta3", "label"))
    for i in range(81):
        x = i / 720
        for j in range(81):
            y = j / 720
            s12 = 1/3 - x + y
            s13 = 1/3 + x - y
            s23 = 1/3 - x - y
            delta23 = 1/9 - 2/3*(x + y) + (x - y)**2
            delta3 = 1/27 - 2/9*(x + y) - (x - y)**2
            writer.writerow((0, x, y, s12, s13, s23, delta23, delta3,
            labeled_points.get((i, j), "")))

from __future__ import annotations

import csv
import json
from pathlib import Path

from matplotlib import rc_context
from matplotlib.figure import Figure
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from PIL import Image


def invert_colorbar(heat_path: Path, bar_path: Path) -> tuple[np.ndarray, int]:
    heat = np.asarray(Image.open(heat_path).convert("RGB"), dtype=float)
    bar = np.asarray(Image.open(bar_path).convert("RGB"), dtype=float)
    if heat.shape != (656, 656, 3) or bar.shape != (1419, 57, 3):
        raise ValueError("Unexpected Preiss Fig. 2(a) raster or colorbar dimensions.")

    colors = np.median(bar[:, 15:45, :], axis=1)
    values = 1.0 - np.arange(len(colors), dtype=float) / (len(colors) - 1)

    unique_colors, inverse = np.unique(colors.astype(np.uint8), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    unique_values = np.asarray(
    [values[inverse == index].mean() for index in range(len(unique_colors))]
    )

    pixels = heat.reshape(-1, 3)
    reconstructed = np.empty(len(pixels), dtype=float)
    chunk = 50_000
    for start in range(0, len(pixels), chunk):
        stop = min(start + chunk, len(pixels))
        distance2 = (
        (pixels[start:stop, None, :] - unique_colors[None, :, :]) ** 2
        ).sum(axis=2)
        reconstructed[start:stop] = unique_values[distance2.argmin(axis=1)]

    grayscale_annotation = (
    (np.ptp(pixels, axis=1) <= 4) & (pixels.mean(axis=1) < 245)
    )
    reconstructed[grayscale_annotation] = np.nan
    intensity = reconstructed.reshape(heat.shape[:2])

    swapped = np.flipud(np.fliplr(intensity)).T
    missing = np.isnan(intensity)
    fillable = missing & ~np.isnan(swapped)
    intensity[fillable] = swapped[fillable]
    if np.isnan(intensity).any():
        raise RuntimeError("Some annotation pixels could not be restored by symmetry.")

    return np.clip(intensity, 0.0, None), int(missing.sum())


def terminal_statistics(
intensity: np.ndarray,
phase_offset: float = 0.0,
axis_limit: float = 3.7,
) -> dict[str, object]:
    height, width = intensity.shape
    q1 = np.linspace(-axis_limit, axis_limit, width, endpoint=False)
    q1 += 2 * axis_limit / (2 * width)
    q2 = np.linspace(axis_limit, -axis_limit, height, endpoint=False)
    q2 -= 2 * axis_limit / (2 * height)

    labels1 = np.floor(np.mod(q1 - phase_offset, 2.0) / (2.0 / 3.0)).astype(int)
    labels2 = np.floor(np.mod(q2 - phase_offset, 2.0) / (2.0 / 3.0)).astype(int)
    labels1 = np.clip(labels1, 0, 2)
    labels2 = np.clip(labels2, 0, 2)

    ordered = np.zeros((3, 3), dtype=float)
    for row in range(3):
        for col in range(3):
            ordered[row, col] = intensity[np.ix_(labels2 == row, labels1 == col)].sum()
    ordered /= ordered.sum()

    p = np.diag(ordered)
    s = np.asarray(
    [
    ordered[0, 1] + ordered[1, 0],
    ordered[0, 2] + ordered[2, 0],
    ordered[1, 2] + ordered[2, 1],
    ]
    )

    q12 = s[0] ** 2 - 4 * p[0] * p[1]
    q13 = s[1] ** 2 - 4 * p[0] * p[2]
    q23 = s[2] ** 2 - 4 * p[1] * p[2]
    cubic = (
    4 * p[0] * p[1] * p[2]
    + s[0] * s[1] * s[2]
    - p[0] * s[2] ** 2
    - p[1] * s[1] ** 2
    - p[2] * s[0] ** 2
    )
    matrix_g = np.asarray(
    [
    [p[0], s[0] / 2, s[1] / 2],
    [s[0] / 2, p[1], s[2] / 2],
    [s[1] / 2, s[2] / 2, p[2]],
    ]
    )
    eigenvalues = np.linalg.eigvalsh(matrix_g)

    return {
    "phase_offset": float(phase_offset),
    "axis_limit": float(axis_limit),
    "ordered_terminal_matrix": ordered.tolist(),
    "p": p.tolist(),
    "s": s.tolist(),
    "Q": [float(q12), float(q13), float(q23)],
    "C": float(cubic),
    "G": matrix_g.tolist(),
    "eigenvalues": eigenvalues.tolist(),
    "passive_criterion_satisfied": bool(
    min(q12, q13, q23, cubic) >= -1e-12
    and np.count_nonzero(eigenvalues > 1e-12) <= 1
    ),
    }


def scan_limits(intensity, limits, offsets):
    margins = np.empty((len(limits), len(offsets), 2), dtype=float)
    for row, limit in enumerate(limits):
        for column, offset in enumerate(offsets):
            result = terminal_statistics(intensity, phase_offset=offset, axis_limit=limit)
            margins[row, column] = result["C"], min(result["Q"])
    return margins


def write_csv(path: Path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def calculate(data_dir: Path, csv_dir: Path) -> dict:
    source = json.loads((data_dir / "preiss_source.json").read_text(encoding="utf-8"))
    intensity, masked_pixels = invert_colorbar(
    data_dir / "preiss_fig2a_native_raster.png",
    data_dir / "preiss_fig2a_colorbar.png",
    )
    primary = terminal_statistics(intensity, phase_offset=0.0, axis_limit=3.7)
    phase_offsets = np.linspace(0.0, 2.0, 601, endpoint=False)
    phase_scan = scan_limits(intensity, [3.7], phase_offsets)[0]
    limits = np.linspace(3.55, 3.85, 31)
    calibration_offsets = np.linspace(0.0, 2.0, 241, endpoint=False)
    calibration_scan = scan_limits(intensity, limits, calibration_offsets)
    summary = {
    "phase_scan_C_min": float(phase_scan[:, 0].min()),
    "phase_scan_C_max": float(phase_scan[:, 0].max()),
    "phase_scan_min_Q": float(phase_scan[:, 1].min()),
    "phase_scan_fraction_passing": float(np.mean(np.all(phase_scan >= 0, axis=1))),
    "axis_and_phase_scan_C_min": float(calibration_scan[:, :, 0].min()),
    "axis_and_phase_scan_min_Q": float(calibration_scan[:, :, 1].min()),
    }
    csv_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv_dir / "preiss_digitized_intensity.csv", intensity, delimiter=",", fmt="%.17g")
    write_csv(
    csv_dir / "preiss_histogram.csv",
    ["phase_offset", "axis_half_width", "p1", "p2", "p3", "s12", "s13", "s23",
    "Delta12", "Delta13", "Delta23", "Delta3"],
    [(primary["phase_offset"], primary["axis_limit"], *primary["p"], *primary["s"],
    *primary["Q"], primary["C"])],
    )
    write_csv(
    csv_dir / "preiss_phase_scan.csv",
    ["phase_offset", "axis_half_width", "Delta3", "min_pairwise_margin"],
    ((offset, 3.7, cubic, min_q)
    for offset, (cubic, min_q) in zip(phase_offsets, phase_scan)),
    )
    write_csv(
    csv_dir / "preiss_calibration_scan.csv",
    ["phase_offset", "axis_half_width", "Delta3", "min_pairwise_margin"],
    ((offset, limit, *calibration_scan[row, column])
    for row, limit in enumerate(limits)
    for column, offset in enumerate(calibration_offsets)),
    )
    write_csv(csv_dir / "preiss_robustness_summary.csv", ["quantity", "value"], summary.items())
    return {
    "source": source,
    "digitization": {
    "raster_shape": list(intensity.shape),
    "masked_annotation_pixels_restored_by_symmetry": masked_pixels,
    "axis_extent_in_k_over_k_lat": [-3.7, 3.7],
    "terminal_definition": "floor(((q-phase_offset) mod 2)/(2/3)), q=k/k_lat",
    },
    "primary_result": primary,
    "robustness": summary,
    "robustness_points": {
    "phase_offsets": phase_offsets.tolist(),
    "phase_cubic": phase_scan[:, 0].tolist(),
    "calibration_limits": limits.tolist(),
    "calibration_cubic": calibration_scan[:, :, 0].tolist(),
    },
    }


def generate_robustness(result: dict, figures_dir: Path) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    points = result["robustness_points"]
    limits = points["calibration_limits"]
    half_step = (limits[1] - limits[0]) / 2
    with rc_context({
    "font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8, "axes.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.direction": "in", "ytick.direction": "in", "text.usetex": False,
    }):
        figure = Figure(figsize=(6.5, 2.25))
        left = figure.add_axes([0.100, 0.220, 0.331, 0.642])
        right = figure.add_axes([0.569, 0.220, 0.305, 0.642])
        color_axis = figure.add_axes([0.882, 0.220, 0.011, 0.642])
        left.plot(points["phase_offsets"], points["phase_cubic"], color="#2864a1",
        linewidth=1.05)
        left.set_title(r"$L=3.7$: origin sensitivity", pad=5)
        left.set_ylabel(r"$\Delta_3$", labelpad=5)
        left.set_ylim(0.0103, 0.0122)
        left.set_yticks([0.0105, 0.0110, 0.0115, 0.0120])
        left.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        minimum = result["robustness"]["phase_scan_C_min"]
        maximum = result["robustness"]["phase_scan_C_max"]
        left.text(0.032, 0.04, rf"${minimum:.5f}\leq\Delta_3\leq{maximum:.5f}$",
        transform=left.transAxes, fontsize=7)

        image = right.imshow(
        points["calibration_cubic"], origin="lower", aspect="auto",
        interpolation="nearest", extent=(0, 2, limits[0] - half_step,
        limits[-1] + half_step), cmap="viridis",
        )
        right.set_title("Axis calibration sensitivity", pad=5)
        right.set_ylabel(r"Axis half-width $L$", labelpad=5)
        right.set_yticks(np.linspace(3.55, 3.85, 7))
        right.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        colorbar = figure.colorbar(image, cax=color_axis, format="%.5f")
        colorbar.set_ticks([0.01050, 0.01075, 0.01100, 0.01125, 0.01150, 0.01175])
        colorbar.set_label(r"$\Delta_3$", labelpad=4)
        for axis in (left, right):
            axis.set_xlim(0, 2)
            axis.set_xticks([0, 1, 2])
            axis.set_xlabel(r"Sector origin $\phi$", labelpad=4)
        figure.text(0.037, 0.925, "(a)", weight="bold")
        figure.text(0.506, 0.925, "(b)", weight="bold")
        path = figures_dir / "SupFig2.png"
        figure.savefig(path, dpi=250, facecolor="white")
    return path

if __name__ == "__main__":
    scripts_dir = Path(__file__).resolve().parent
    result = calculate(scripts_dir.parent / "data", scripts_dir)
    generate_robustness(result, scripts_dir)

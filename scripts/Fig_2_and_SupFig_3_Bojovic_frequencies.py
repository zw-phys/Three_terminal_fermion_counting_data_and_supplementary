from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "data" / "bojovic_fig5e.csv"
OUTPUT_STATES = ("0_updown", "updown_0", "up_down", "down_up")
OUTCOMES = ("200", "020", "002", "110", "101", "011")


def run(output_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    frequencies: dict[int, dict[str, tuple[Fraction, float, float]]] = {}
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            pulse = int(row["x"])
            state = row["OutputState"]
            if row["InputState"] != "updown_0" or state not in OUTPUT_STATES:
                raise ValueError(f"Unexpected state at workbook row {row['sheet_row']}.")
            outcomes = frequencies.setdefault(pulse, {})
            if state in outcomes:
                raise ValueError(f"Duplicate state {state} at x={pulse}.")
            lower, upper = float(row["yerr_lower"]), float(row["yerr_upper"])
            frequency = Fraction(row["y"])
            if not (0 <= frequency <= 1 and 0 <= lower <= 1 and 0 <= upper <= 1):
                raise ValueError(f"Invalid frequency or interval at x={pulse}, {state}.")
            outcomes[state] = frequency, lower, upper

    if any(set(outcomes) != set(OUTPUT_STATES) for outcomes in frequencies.values()):
        raise ValueError("Expected four resolved output states at each pulse setting.")
    totals = {
    pulse: sum(frequency for frequency, _, _ in outcomes.values())
    for pulse, outcomes in frequencies.items()
    }
    complete = sorted(pulse for pulse, total in totals.items() if total == 1)
    if complete != [30, 50]:
        raise ValueError(f"Expected complete histograms at x=30 and 50, got {complete}.")

    probabilities, errors, mapped_rows = [], [], []
    for pulse in complete:
        row = [(Fraction(0), 0.0, 0.0)] * 2
        for state in OUTPUT_STATES:
            published, lower, upper = frequencies[pulse][state]
            frequency = published.limit_denominator(100)
            if abs(frequency - published) > Fraction(1, 2_000_000_000):
                raise ValueError(f"Frequency reconstruction exceeds rounding at x={pulse}.")
            row.append((frequency, lower, upper))
        if sum(frequency for frequency, _, _ in row) != 1:
            raise ValueError(f"Reconstructed probabilities do not sum to one at x={pulse}.")
        probabilities.append([float(frequency) for frequency, _, _ in row])
        lower_errors = [lower for _, lower, _ in row[2:]]
        upper_errors = [upper for _, _, upper in row[2:]]
        errors.append([lower_errors, upper_errors])
        for outcome, (frequency, lower, upper) in zip(OUTCOMES, row):
            mapped_rows.append([pulse, outcome, str(frequency), float(frequency), lower, upper])

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "bojovic_terminal_probabilities.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
            "pulse_us", "outcome", "frequency_fraction", "probability",
            "error_lower_68", "error_upper_68",
            ])
            writer.writerows(mapped_rows)

    return np.asarray(probabilities), np.asarray(errors)


if __name__ == "__main__":
    run(Path(__file__).resolve().parent)

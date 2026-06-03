import csv
import math
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("This plotter needs matplotlib. Install it with: python3 -m pip install matplotlib")
    raise SystemExit(1)


def parse_float(value: str) -> float:
    value = value.strip()
    if not value:
        return float("nan")
    return float(value)


def load_rows(csv_path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        fieldnames = reader.fieldnames or []
        rows: list[dict[str, float]] = []
        for raw_row in reader:
            row: dict[str, float] = {}
            for field in fieldnames:
                row[field] = parse_float(raw_row.get(field, ""))
            rows.append(row)
    return fieldnames, rows


def parse_arguments(argv: list[str]) -> Path | None:
    if len(argv) == 2:
        return Path(argv[1]).expanduser()

    if len(argv) == 3 and argv[1] in {"--show", "--save"}:
        return Path(argv[2]).expanduser()

    print(
        "Usage: python3 Portable/scripts/plot_delay_courbe_detector.py "
        "<file.csv>"
    )
    return None


def finalize_figure(figure, csv_path: Path) -> Path:
    output_path = csv_path.with_suffix(".png")
    figure.savefig(output_path, dpi=160)
    plt.show()
    plt.close(figure)
    return output_path


def plot_signal(csv_path: Path, rows: list[dict[str, float]]):
    time_ms = [row["time_ms"] for row in rows]
    reference = [row["reference_output"] for row in rows]
    captured = [row["input_ch1"] for row in rows]
    delay_ms = rows[0]["delay_ms_marker"] if rows else float("nan")

    figure, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_ms, reference, label="reference_output", linewidth=2)
    ax.plot(time_ms, captured, label="input_ch1", linewidth=1.5)
    if not math.isnan(delay_ms):
        ax.axvline(
            delay_ms,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"x = delay = {delay_ms:.3f} ms",
        )
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude")
    ax.set_title(csv_path.stem)
    ax.grid(True, alpha=0.3)
    ax.legend()
    figure.tight_layout()
    return figure


def plot_distribution(csv_path: Path, rows: list[dict[str, float]]):
    delays_ms = [
        row["delay_ms"]
        for row in rows
        if not math.isnan(row["delay_ms"])
    ]
    if not delays_ms:
        raise ValueError("No valid delay_ms rows found for histogram.")

    figure, ax = plt.subplots(figsize=(9, 5))
    ax.hist(delays_ms, bins=100, edgecolor="black")
    ax.set_xlabel("delay (ms)")
    ax.set_ylabel("count")
    ax.set_title(csv_path.stem)
    ax.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def finite_values(rows: list[dict[str, float]], field: str) -> list[float]:
    return [
        row[field]
        for row in rows
        if field in row and not math.isnan(row[field])
    ]


def plot_delay_comparison(csv_path: Path, rows: list[dict[str, float]]):
    repetitions = [
        row["repetition"]
        for row in rows
        if not math.isnan(row["repetition"])
    ]
    raw_delays = [
        row["raw_delay_ms"]
        for row in rows
        if not math.isnan(row["repetition"])
    ]
    corrected_delays = [
        row["timestamp_corrected_delay_ms"]
        for row in rows
        if not math.isnan(row["repetition"])
    ]
    delta_delays = finite_values(rows, "delay_correction_delta_ms")

    if not repetitions:
        raise ValueError("No valid repetition rows found for delay comparison.")

    figure, (ax_series, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=False,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )

    ax_series.plot(
        repetitions,
        raw_delays,
        color="#2563eb",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        label="raw delay",
    )
    ax_series.plot(
        repetitions,
        corrected_delays,
        color="#dc2626",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        label="PortAudio-corrected delay",
    )
    ax_series.set_xlabel("repetition")
    ax_series.set_ylabel("delay (ms)")
    ax_series.set_title(csv_path.stem)
    ax_series.grid(True, alpha=0.3)
    ax_series.legend()

    if delta_delays:
        ax_hist.hist(delta_delays, bins=min(60, max(10, len(delta_delays) // 2)), color="#7c3aed", edgecolor="black")
        ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax_hist.set_xlabel("corrected - raw delay (ms)")
        ax_hist.set_ylabel("count")
        ax_hist.grid(True, axis="y", alpha=0.3)
    else:
        ax_hist.text(
            0.5,
            0.5,
            "No finite correction deltas available",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
        )
        ax_hist.set_axis_off()

    figure.tight_layout()
    return figure


def main() -> int:
    csv_path = parse_arguments(sys.argv)
    if csv_path is None:
        return 1

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    fieldnames, rows = load_rows(csv_path)
    if not rows:
        print(f"No rows found in {csv_path}")
        return 1

    field_set = set(fieldnames)
    if {"time_ms", "reference_output", "input_ch1", "delay_ms_marker"}.issubset(field_set):
        figure = plot_signal(csv_path, rows)
    elif {
        "repetition",
        "raw_delay_ms",
        "timestamp_corrected_delay_ms",
        "delay_correction_delta_ms",
    }.issubset(field_set):
        figure = plot_delay_comparison(csv_path, rows)
    elif {"repetition", "delay_ms"}.issubset(field_set):
        figure = plot_distribution(csv_path, rows)
    else:
        print(f"Unsupported CSV shape for {csv_path}")
        return 1

    output_path = finalize_figure(figure, csv_path)
    print(f"Saved plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

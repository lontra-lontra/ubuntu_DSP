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

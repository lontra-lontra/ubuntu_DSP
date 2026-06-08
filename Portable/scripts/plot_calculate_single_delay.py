import csv
import math
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("This plotter needs matplotlib. Install it with: python3 -m pip install matplotlib")
    raise SystemExit(1)


REFERENCE_COLOR = "#111827"
CAPTURED_COLOR = "#0f766e"
ALIGNED_COLOR = "#b45309"


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

    print("Usage: python3 Portable/scripts/plot_calculate_single_delay.py <file.csv>")
    return None


def first_value(rows: list[dict[str, float]], field: str) -> float:
    if not rows:
        return float("nan")
    return rows[0].get(field, float("nan"))


def plot_capture(csv_path: Path, rows: list[dict[str, float]]):
    time_ms = [row["time_ms"] for row in rows]
    reference_output = [row["reference_output"] for row in rows]
    captured_input = [row["captured_input"] for row in rows]
    aligned_input = [row["aligned_input"] for row in rows]

    delay_samples = first_value(rows, "correlation_delay_samples")
    delay_ms = first_value(rows, "correlation_delay_ms")
    score = first_value(rows, "correlation_score")

    figure, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(
        time_ms,
        reference_output,
        color=REFERENCE_COLOR,
        linewidth=1.6,
        label="reference_output",
    )
    ax.plot(
        time_ms,
        captured_input,
        color=CAPTURED_COLOR,
        linewidth=1.2,
        alpha=0.85,
        label="captured_input",
    )
    ax.plot(
        time_ms,
        aligned_input,
        color=ALIGNED_COLOR,
        linewidth=1.2,
        alpha=0.9,
        label="aligned_input",
    )

    title_bits = [csv_path.stem]
    if not math.isnan(delay_samples):
        title_bits.append(f"delay={delay_samples:.0f} samples")
    if not math.isnan(delay_ms):
        title_bits.append(f"{delay_ms:.3f} ms")
    if not math.isnan(score):
        title_bits.append(f"corr={score:.4f}")

    ax.set_title(" | ".join(title_bits))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figure.tight_layout()
    return figure


def main() -> int:
    csv_path = parse_arguments(sys.argv)
    if csv_path is None:
        return 1
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    _, rows = load_rows(csv_path)
    if not rows:
        print(f"CSV has no data rows: {csv_path}")
        return 1

    figure = plot_capture(csv_path, rows)
    output_path = csv_path.with_suffix(".png")
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.show()
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

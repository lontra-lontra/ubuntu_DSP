import csv
import math
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("This plotter needs matplotlib. Install it with: python3 -m pip install matplotlib")
    raise SystemExit(1)


REFERENCE_COLOR = "#ea580c"
RESPONSE_COLOR = "#0f766e"
RESPONSE_PALETTE = [
    "#0f766e",
    "#2563eb",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#ca8a04",
    "#db2777",
    "#16a34a",
    "#ea580c",
    "#1d4ed8",
]


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


def plot_latest(csv_path: Path, rows: list[dict[str, float]]) -> Path:
    time_ms = [row["time_ms"] for row in rows]
    reference = [row["normalized_reference"] for row in rows]
    response = [row["normalized_response"] for row in rows]

    figure, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        time_ms,
        reference,
        linewidth=2.2,
        color=REFERENCE_COLOR,
    )
    ax.plot(
        time_ms,
        response,
        linewidth=1.7,
        color=RESPONSE_COLOR,
    )
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized amplitude")
    ax.set_title(csv_path.stem)
    ax.grid(True, alpha=0.3)
    figure.tight_layout()

    output_path = csv_path.with_suffix(".png")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_consolidated(
    csv_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, float]],
) -> Path:
    time_ms = [row["time_ms"] for row in rows]
    reference = [row["normalized_reference"] for row in rows]
    response_fields = [
        field for field in fieldnames if field.startswith("response_rep_")
    ]

    figure, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        time_ms,
        reference,
        linewidth=2.5,
        color=REFERENCE_COLOR,
    )

    if response_fields:
        alpha = min(0.7, max(0.14, 6.0 / float(len(response_fields))))
        for index, field in enumerate(response_fields):
            response = [row[field] for row in rows]
            ax.plot(
                time_ms,
                response,
                linewidth=1.15,
                alpha=alpha,
                color=RESPONSE_PALETTE[index % len(RESPONSE_PALETTE)],
            )

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized amplitude")
    ax.set_title(csv_path.stem)
    ax.grid(True, alpha=0.3)
    figure.tight_layout()

    output_path = csv_path.with_suffix(".png")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 Portable/scripts/plot_superpose_signals_and_responses.py <file.csv>")
        return 1

    csv_path = Path(sys.argv[1]).expanduser()
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    fieldnames, rows = load_rows(csv_path)
    if not rows:
        print(f"No rows found in {csv_path}")
        return 1

    field_set = set(fieldnames)
    if {"time_ms", "normalized_reference", "normalized_response"}.issubset(field_set):
        output_path = plot_latest(csv_path, rows)
    elif "time_ms" in field_set and "normalized_reference" in field_set:
        output_path = plot_consolidated(csv_path, fieldnames, rows)
    else:
        print(f"Unsupported CSV shape for {csv_path}")
        return 1

    print(f"Saved plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

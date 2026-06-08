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


def resolve_field_name(fieldnames: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def finite_pairs(
    rows: list[dict[str, float]],
    x_field: str,
    y_field: str,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        x_value = row.get(x_field, float("nan"))
        y_value = row.get(y_field, float("nan"))
        if math.isnan(x_value) or math.isnan(y_value):
            continue
        x_values.append(x_value)
        y_values.append(y_value)
    return x_values, y_values


def histogram_bin_count(values: list[float]) -> int:
    return min(40, max(10, len(values) // 2))


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return float("nan")

    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    numerator = 0.0
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    for x_value, y_value in zip(x_values, y_values):
        dx = x_value - mean_x
        dy = y_value - mean_y
        numerator += dx * dy
        sum_sq_x += dx * dx
        sum_sq_y += dy * dy

    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def plot_delay_comparison(
    csv_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, float]],
):
    threshold_field = resolve_field_name(fieldnames, ["threshold_delay_ms", "raw_delay_ms"])
    corrected_field = resolve_field_name(fieldnames, ["timestamp_corrected_delay_ms"])
    correlation_field = resolve_field_name(fieldnames, ["correlation_delay_ms"])
    first_callback_field = resolve_field_name(fieldnames, ["first_output_minus_input_ms"])

    if (
        threshold_field is None or
        corrected_field is None or
        correlation_field is None or
        first_callback_field is None
    ):
        raise ValueError("Missing threshold, corrected, correlation, or first callback delay fields.")

    threshold_delays = finite_values(rows, threshold_field)
    corrected_delays = finite_values(rows, corrected_field)
    scatter_x, scatter_y = finite_pairs(rows, first_callback_field, correlation_field)

    if not threshold_delays and not corrected_delays and not scatter_x:
        raise ValueError("No finite threshold, corrected, or correlation delays found.")

    figure = plt.figure(figsize=(13, 8))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.2])
    ax_threshold = figure.add_subplot(grid[0, 0])
    ax_corrected = figure.add_subplot(grid[0, 1])
    ax_scatter = figure.add_subplot(grid[1, :])

    if threshold_delays:
        ax_threshold.hist(
            threshold_delays,
            bins=histogram_bin_count(threshold_delays),
            color="#2563eb",
            alpha=0.8,
            edgecolor="black",
        )
    ax_threshold.set_xlabel("threshold delay (ms)")
    ax_threshold.set_ylabel("count")
    ax_threshold.set_title("Threshold Delay Distribution")
    ax_threshold.grid(True, axis="y", alpha=0.3)

    if corrected_delays:
        ax_corrected.hist(
            corrected_delays,
            bins=histogram_bin_count(corrected_delays),
            color="#dc2626",
            alpha=0.8,
            edgecolor="black",
        )
    ax_corrected.set_xlabel("timestamp-corrected delay (ms)")
    ax_corrected.set_ylabel("count")
    ax_corrected.set_title("Corrected Delay Distribution")
    ax_corrected.grid(True, axis="y", alpha=0.3)

    if scatter_x and scatter_y:
        ax_scatter.scatter(
            scatter_x,
            scatter_y,
            color="#0f766e",
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            label=f"repetitions ({len(scatter_x)})",
        )
        correlation_value = pearson_correlation(scatter_x, scatter_y)
        if not math.isnan(correlation_value):
            ax_scatter.set_title(
                f"Correlation Delay vs First Output Minus Input (r = {correlation_value:.3f})"
            )
        else:
            ax_scatter.set_title("Correlation Delay vs First Output Minus Input")
        ax_scatter.legend()
    else:
        ax_scatter.text(
            0.5,
            0.5,
            "No finite first-callback/correlation delay pairs found.",
            ha="center",
            va="center",
            transform=ax_scatter.transAxes,
        )
        ax_scatter.set_title("Correlation Delay vs First Output Minus Input")

    ax_scatter.set_xlabel("first output minus input (ms)")
    ax_scatter.set_ylabel("correlation delay (ms)")
    ax_scatter.grid(True, alpha=0.3)
    figure.suptitle(csv_path.stem)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
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
        "timestamp_corrected_delay_ms",
    }.issubset(field_set):
        figure = plot_delay_comparison(csv_path, fieldnames, rows)
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

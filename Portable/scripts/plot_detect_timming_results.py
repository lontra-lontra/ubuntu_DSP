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


def load_results(csv_path: Path) -> list[dict[str, float | list[float]]]:
    rows: list[dict[str, float | list[float]]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        fieldnames = reader.fieldnames or []
        if "delay_input0_to_input1_burst_1_samples" in fieldnames:
            delay_sample_keys = [
                "delay_input0_to_input1_burst_1_samples",
                "delay_input0_to_input1_burst_2_samples",
                "delay_input0_to_input1_burst_3_samples",
            ]
        else:
            delay_sample_keys = [
                "delay_burst_1_samples",
                "delay_burst_2_samples",
                "delay_burst_3_samples",
            ]

        for raw_row in reader:
            sample_rate = parse_float(raw_row["sample_rate_hz"])
            buffer_size = parse_float(raw_row["buffer_size_samples"])
            delay_samples = [
                parse_float(raw_row[delay_sample_keys[0]]),
                parse_float(raw_row[delay_sample_keys[1]]),
                parse_float(raw_row[delay_sample_keys[2]]),
            ]
            valid_delay_samples = [
                value for value in delay_samples if not math.isnan(value) and value >= 0.0
            ]
            if not valid_delay_samples:
                continue

            mean_delay_samples = sum(valid_delay_samples) / len(valid_delay_samples)
            mean_delay_seconds = mean_delay_samples / sample_rate
            buffer_duration_seconds = buffer_size / sample_rate

            rows.append(
                {
                    "sample_rate_hz": sample_rate,
                    "buffer_size_samples": buffer_size,
                    "valid_burst_count": float(len(valid_delay_samples)),
                    "mean_delay_samples": mean_delay_samples,
                    "mean_delay_seconds": mean_delay_seconds,
                    "mean_delay_ms": mean_delay_seconds * 1000.0,
                    "buffer_duration_seconds": buffer_duration_seconds,
                    "buffer_duration_ms": buffer_duration_seconds * 1000.0,
                    "delay_in_buffers": mean_delay_samples / buffer_size,
                }
            )
    return rows


def group_by_sample_rate(rows: list[dict[str, float | list[float]]]) -> dict[float, list[dict[str, float | list[float]]]]:
    grouped: dict[float, list[dict[str, float | list[float]]]] = {}
    for row in rows:
        sample_rate = float(row["sample_rate_hz"])
        grouped.setdefault(sample_rate, []).append(row)

    for sample_rate_rows in grouped.values():
        sample_rate_rows.sort(key=lambda item: float(item["buffer_size_samples"]))

    return grouped


def save_summary_csv(rows: list[dict[str, float | list[float]]], output_path: Path) -> None:
    fieldnames = [
        "sample_rate_hz",
        "buffer_size_samples",
        "valid_burst_count",
        "mean_delay_samples",
        "mean_delay_seconds",
        "mean_delay_ms",
        "buffer_duration_seconds",
        "buffer_duration_ms",
        "delay_in_buffers",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def plot_delay_vs_buffer_over_frequency(rows: list[dict[str, float | list[float]]], output_path: Path) -> None:
    grouped = group_by_sample_rate(rows)
    figure, ax = plt.subplots(figsize=(9, 6))

    for sample_rate, series in sorted(grouped.items()):
        x_values = [float(row["buffer_duration_ms"]) for row in series]
        y_values = [float(row["mean_delay_ms"]) for row in series]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=f"{sample_rate:.0f} Hz")

    max_x = max(float(row["buffer_duration_ms"]) for row in rows)
    reference = [0.0, max_x]
    ax.plot(reference, reference, linestyle="--", color="black", alpha=0.5, label="y = x")
    ax.set_xlabel("buffer_size / frequency (ms)")
    ax.set_ylabel("mean relay delay (ms)")
    ax.set_title("Relay Delay vs Buffer Duration")
    ax.grid(True, alpha=0.3)
    ax.legend(title="sample rate")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_delay_samples_vs_buffer_size(rows: list[dict[str, float | list[float]]], output_path: Path) -> None:
    grouped = group_by_sample_rate(rows)
    figure, ax = plt.subplots(figsize=(9, 6))

    for sample_rate, series in sorted(grouped.items()):
        x_values = [float(row["buffer_size_samples"]) for row in series]
        y_values = [float(row["mean_delay_samples"]) for row in series]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=f"{sample_rate:.0f} Hz")

    ax.set_xlabel("buffer size (samples)")
    ax.set_ylabel("mean relay delay (samples)")
    ax.set_title("Relay Delay (Samples) vs Buffer Size")
    ax.grid(True, alpha=0.3)
    ax.legend(title="sample rate")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_delay_in_buffers(rows: list[dict[str, float | list[float]]], output_path: Path) -> None:
    grouped = group_by_sample_rate(rows)
    figure, ax = plt.subplots(figsize=(9, 6))

    for sample_rate, series in sorted(grouped.items()):
        x_values = [float(row["buffer_size_samples"]) for row in series]
        y_values = [float(row["delay_in_buffers"]) for row in series]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=f"{sample_rate:.0f} Hz")

    ax.set_xlabel("buffer size (samples)")
    ax.set_ylabel("mean relay delay / buffer size")
    ax.set_title("Insight: Relay Delay Expressed in Buffer Lengths")
    ax.grid(True, alpha=0.3)
    ax.legend(title="sample rate")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: py -3 Portable/scripts/plot_detect_timming_results.py Portable/output/detect_timming_results.csv")
        return 1

    csv_path = Path(sys.argv[1]).expanduser()
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    rows = load_results(csv_path)
    if not rows:
        print("No valid timing rows found in CSV.")
        return 1

    output_dir = csv_path.parent
    summary_csv_path = output_dir / "detect_timming_results_summary.csv"
    plot1_path = output_dir / "detect_timming_delay_vs_buffer_over_frequency.png"
    plot2_path = output_dir / "detect_timming_delay_samples_vs_buffer_size.png"
    plot3_path = output_dir / "detect_timming_delay_in_buffers_vs_buffer_size.png"

    save_summary_csv(rows, summary_csv_path)
    plot_delay_vs_buffer_over_frequency(rows, plot1_path)
    plot_delay_samples_vs_buffer_size(rows, plot2_path)
    plot_delay_in_buffers(rows, plot3_path)

    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved plot: {plot1_path}")
    print(f"Saved plot: {plot2_path}")
    print(f"Saved plot: {plot3_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

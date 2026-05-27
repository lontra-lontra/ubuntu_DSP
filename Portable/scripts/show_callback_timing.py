from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("This viewer needs numpy. Install it with: python3 -m pip install numpy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("This viewer needs matplotlib. Install it with: python3 -m pip install matplotlib")
    sys.exit(1)


REQUIRED_COLUMNS = (
    "callback index",
    "callback time (s)",
    "callback delay (s)",
    "allowed delay (s)",
    "callback load (%)",
)


def read_callback_timing_csv(csv_path: str | Path):
    resolved_path = Path(csv_path).expanduser()
    with resolved_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, None)
        if not header:
            raise ValueError("CSV is empty.")

        column_indices = {}
        for required_name in REQUIRED_COLUMNS:
            if required_name not in header:
                raise ValueError(f"CSV is missing required column: {required_name}")
            column_indices[required_name] = header.index(required_name)

        callback_index = []
        callback_time_seconds = []
        callback_delay_seconds = []
        allowed_delay_seconds = []
        callback_load_percent = []

        for row in reader:
            if not row:
                continue

            def read_float(column_name: str) -> float:
                column_index = column_indices[column_name]
                if column_index >= len(row) or not row[column_index].strip():
                    return float("nan")
                return float(row[column_index])

            callback_index.append(read_float("callback index"))
            callback_time_seconds.append(read_float("callback time (s)"))
            callback_delay_seconds.append(read_float("callback delay (s)"))
            allowed_delay_seconds.append(read_float("allowed delay (s)"))
            callback_load_percent.append(read_float("callback load (%)"))

    return {
        "callback_index": np.asarray(callback_index, dtype=float),
        "callback_time_seconds": np.asarray(callback_time_seconds, dtype=float),
        "callback_delay_seconds": np.asarray(callback_delay_seconds, dtype=float),
        "allowed_delay_seconds": np.asarray(allowed_delay_seconds, dtype=float),
        "callback_load_percent": np.asarray(callback_load_percent, dtype=float),
        "path": resolved_path,
    }


def plot_callback_timing(csv_path: str | Path, *, show: bool = True):
    data = read_callback_timing_csv(csv_path)

    callback_time_seconds = data["callback_time_seconds"]
    callback_delay_seconds = data["callback_delay_seconds"]
    allowed_delay_seconds = data["allowed_delay_seconds"]
    callback_load_percent = data["callback_load_percent"]

    if callback_delay_seconds.size == 0:
        raise ValueError("CSV contains no callback timing rows.")

    finite_allowed = allowed_delay_seconds[np.isfinite(allowed_delay_seconds)]
    representative_allowed_delay = (
        float(np.nanmedian(finite_allowed))
        if finite_allowed.size > 0
        else float("nan")
    )

    plt.rcParams.update(
        {
            "axes.facecolor": "#fcfaf7",
            "figure.facecolor": "#f5f1ea",
            "axes.edgecolor": "#3f3b37",
            "axes.labelcolor": "#2c2622",
            "xtick.color": "#2c2622",
            "ytick.color": "#2c2622",
            "text.color": "#2c2622",
            "font.size": 10,
        }
    )

    figure, (ax_time, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(13.5, 8.0),
        constrained_layout=True,
    )

    ax_time.plot(
        callback_time_seconds,
        callback_delay_seconds,
        color="#1d4ed8",
        linewidth=1.7,
        marker="o",
        markersize=3.0,
        label="callback delay",
    )
    ax_time.plot(
        callback_time_seconds,
        allowed_delay_seconds,
        color="#dc2626",
        linewidth=1.5,
        linestyle="--",
        label="allowed delay",
    )
    ax_time.set_title("Callback delay vs time")
    ax_time.set_xlabel("Callback time (s)")
    ax_time.set_ylabel("Delay (s)")
    ax_time.grid(True, alpha=0.25)
    ax_time.legend(loc="upper right")

    finite_delays = callback_delay_seconds[np.isfinite(callback_delay_seconds)]
    bin_count = int(np.clip(np.sqrt(max(finite_delays.size, 1)), 10, 60))
    ax_hist.hist(
        finite_delays,
        bins=bin_count,
        color="#0f766e",
        alpha=0.75,
        edgecolor="#134e4a",
    )
    if np.isfinite(representative_allowed_delay):
        ax_hist.axvline(
            representative_allowed_delay,
            color="#dc2626",
            linestyle="--",
            linewidth=1.8,
            label="allowed delay",
        )
    ax_hist.set_title("Callback delay distribution")
    ax_hist.set_xlabel("Delay (s)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, alpha=0.2)
    if np.isfinite(representative_allowed_delay):
        ax_hist.legend(loc="upper right")

    figure.suptitle(
        f"Callback timing viewer: {data['path'].name}",
        fontsize=13,
        fontweight="semibold",
    )

    if show:
        plt.show()

    return figure, (ax_time, ax_hist)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: py -3 Portable/scripts/show_callback_timing.py <callback_timing.csv>")
        return 1

    csv_argument = Path(sys.argv[1]).expanduser()
    if not csv_argument.exists():
        print(f"File not found: {csv_argument}")
        return 1

    try:
        plot_callback_timing(csv_argument, show=True)
    except ValueError as ex:
        print(ex)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

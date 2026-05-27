from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    print("This viewer needs numpy. Install it with: python3 -m pip install numpy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    print("This viewer needs matplotlib. Install it with: python3 -m pip install matplotlib")
    sys.exit(1)


HEADER_RE = re.compile(r"^(input|output|expected output) (\d+)$")


def get_legend_handles(legend):
    handles = getattr(legend, "legendHandles", None)
    if handles is not None:
        return handles
    handles = getattr(legend, "legend_handles", None)
    if handles is not None:
        return handles
    return []


def parse_validation_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resolved_path = Path(csv_path).expanduser()
    with resolved_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, None)
        if not header or len(header) < 4:
            raise ValueError("CSV is empty or missing required columns.")
        if header[0].strip() != "time (s)":
            raise ValueError("First column must be 'time (s)'.")

        grouped_columns: dict[int, dict[str, int]] = {}
        for column_index, column_name in enumerate(header[1:], start=1):
            match = HEADER_RE.match(column_name.strip())
            if not match:
                continue
            role = match.group(1)
            channel = int(match.group(2))
            grouped_columns.setdefault(channel, {})[role] = column_index

        if not grouped_columns:
            raise ValueError("Could not find any input/output/expected output channel groups.")

        channel_indices = sorted(grouped_columns)
        for channel in channel_indices:
            roles = grouped_columns[channel]
            if "input" not in roles or "output" not in roles or "expected output" not in roles:
                raise ValueError(f"Channel {channel} is missing one of input/output/expected output.")

        rows = list(reader)
        sample_count = len(rows)
        channel_count = len(channel_indices)

        time_axis = np.full(sample_count, np.nan, dtype=float)
        inputs = np.full((channel_count, sample_count), np.nan, dtype=float)
        outputs = np.full((channel_count, sample_count), np.nan, dtype=float)
        expected_outputs = np.full((channel_count, sample_count), np.nan, dtype=float)

        for row_index, row in enumerate(rows):
            if row and row[0].strip():
                time_axis[row_index] = float(row[0])

            for normalized_channel, original_channel in enumerate(channel_indices):
                roles = grouped_columns[original_channel]
                for role, target_array in (
                    ("input", inputs),
                    ("output", outputs),
                    ("expected output", expected_outputs),
                ):
                    column_index = roles[role]
                    if column_index < len(row) and row[column_index].strip():
                        target_array[normalized_channel, row_index] = float(row[column_index])

        return time_axis, inputs, outputs, expected_outputs


def compute_summary_matrix(
    outputs: np.ndarray,
    expected_outputs: np.ndarray,
) -> np.ndarray:
    abs_diff = np.abs(outputs - expected_outputs)
    expected_peak = np.nanmax(np.abs(expected_outputs), axis=1)
    expected_peak = np.maximum(expected_peak, 1e-12)
    max_abs_diff = np.nanmax(abs_diff, axis=1)
    relative_error_percent = 100.0 * max_abs_diff / expected_peak
    return relative_error_percent.reshape(1, -1)


def draw_selected_channel(
    figure,
    ax_heatmap,
    ax_signals,
    ax_error,
    highlight,
    *,
    channel_index: int,
    time_axis: np.ndarray,
    inputs: np.ndarray,
    outputs: np.ndarray,
    expected_outputs: np.ndarray,
    summary_matrix: np.ndarray,
) -> None:
    input_signal = inputs[channel_index]
    output_signal = outputs[channel_index]
    expected_signal = expected_outputs[channel_index]
    abs_diff = np.abs(output_signal - expected_signal)

    highlight.set_xy((channel_index - 0.5, -0.5))

    ax_signals.clear()
    ax_error.clear()

    input_line, = ax_signals.plot(
        time_axis,
        input_signal,
        label="input",
        color="#0f766e",
        linewidth=1.8,
    )
    output_line, = ax_signals.plot(
        time_axis,
        output_signal,
        label="output",
        color="#1d4ed8",
        linewidth=1.8,
    )
    expected_line, = ax_signals.plot(
        time_axis,
        expected_signal,
        label="expected output",
        color="#c2410c",
        linewidth=1.6,
        linestyle="--",
    )
    ax_signals.set_title(f"Channel {channel_index}")
    ax_signals.set_ylabel("Amplitude")
    ax_signals.grid(True, alpha=0.25)
    legend = ax_signals.legend(loc="upper right")

    ax_error.plot(time_axis, output_signal - expected_signal, color="#7c3aed", linewidth=1.5)
    ax_error.fill_between(time_axis, 0.0, abs_diff, color="#f97316", alpha=0.18, label="abs diff")
    ax_error.set_xlabel("Time (s)")
    ax_error.set_ylabel("Output - Expected")
    ax_error.set_title("Error trace")
    ax_error.grid(True, alpha=0.25)

    figure._signal_toggle_targets = {}
    signal_lines = [input_line, output_line, expected_line]
    legend_handles = get_legend_handles(legend)
    for legend_handle, signal_line in zip(legend_handles, signal_lines):
        legend_handle.set_picker(True)
        legend_handle.set_pickradius(8)
        figure._signal_toggle_targets[legend_handle] = signal_line

    for legend_text, signal_line in zip(legend.get_texts(), signal_lines):
        legend_text.set_picker(True)
        figure._signal_toggle_targets[legend_text] = signal_line

    for legend_handle, signal_line in zip(legend_handles, signal_lines):
        alpha = 1.0 if signal_line.get_visible() else 0.25
        legend_handle.set_alpha(alpha)

    figure.canvas.draw_idle()


def plot_exact_convolution_validation(
    csv_path: str | Path,
    *,
    show: bool = True,
    selected_channel: int | None = None,
):
    resolved_path = Path(csv_path).expanduser()
    time_axis, inputs, outputs, expected_outputs = parse_validation_csv(resolved_path)
    summary_matrix = compute_summary_matrix(outputs, expected_outputs)
    channel_count = summary_matrix.shape[1]

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

    figure = plt.figure(figsize=(14.0, 8.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.0, 1.6], height_ratios=[1.25, 0.9])
    ax_heatmap = figure.add_subplot(grid[:, 0])
    ax_signals = figure.add_subplot(grid[0, 1])
    ax_error = figure.add_subplot(grid[1, 1], sharex=ax_signals)

    heatmap = ax_heatmap.imshow(summary_matrix, origin="lower", cmap="magma", aspect="auto")
    colorbar = figure.colorbar(heatmap, ax=ax_heatmap, pad=0.03, shrink=0.86)
    colorbar.set_label("Max peak-relative error (%)")

    ax_heatmap.set_xticks(np.arange(channel_count))
    ax_heatmap.set_xticklabels([str(index) for index in range(channel_count)])
    ax_heatmap.set_yticks([0])
    ax_heatmap.set_yticklabels(["validation"])
    ax_heatmap.set_xlabel("Channel")
    ax_heatmap.set_title("Exact convolution check\n(click a channel to inspect traces)")
    ax_heatmap.set_xticks(np.arange(-0.5, channel_count, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, 1.5, 1), minor=True)
    ax_heatmap.grid(which="minor", color="#ffffff", linewidth=0.5, alpha=0.3)
    ax_heatmap.tick_params(which="minor", bottom=False, left=False)

    for channel in range(channel_count):
        ax_heatmap.text(
            channel,
            0,
            f"{summary_matrix[0, channel]:.4f}%",
            ha="center",
            va="center",
            color="#fffaf2",
            fontsize=9,
            fontweight="semibold",
        )

    highlight = Rectangle(
        (-0.5, -0.5),
        1.0,
        1.0,
        fill=False,
        linewidth=2.4,
        edgecolor="#22c55e",
    )
    ax_heatmap.add_patch(highlight)

    def on_click(event) -> None:
        if event.inaxes is not ax_heatmap or event.xdata is None:
            return

        channel_index = int(round(event.xdata))
        if not (0 <= channel_index < channel_count):
            return

        draw_selected_channel(
            figure,
            ax_heatmap,
            ax_signals,
            ax_error,
            highlight,
            channel_index=channel_index,
            time_axis=time_axis,
            inputs=inputs,
            outputs=outputs,
            expected_outputs=expected_outputs,
            summary_matrix=summary_matrix,
        )

    def on_pick(event) -> None:
        toggle_targets = getattr(figure, "_signal_toggle_targets", {})
        signal_line = toggle_targets.get(event.artist)
        if signal_line is None:
            return

        new_visibility = not signal_line.get_visible()
        signal_line.set_visible(new_visibility)

        legend = ax_signals.get_legend()
        if legend is not None:
            legend_handles = get_legend_handles(legend)
            for legend_handle, plotted_line in zip(legend_handles, ax_signals.get_lines()):
                alpha = 1.0 if plotted_line.get_visible() else 0.25
                legend_handle.set_alpha(alpha)
            for legend_text, plotted_line in zip(legend.get_texts(), ax_signals.get_lines()):
                alpha = 1.0 if plotted_line.get_visible() else 0.25
                legend_text.set_alpha(alpha)

        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("pick_event", on_pick)

    initial_channel = int(np.nanargmax(summary_matrix)) if selected_channel is None else selected_channel
    on_click(
        SimpleNamespace(
            inaxes=ax_heatmap,
            xdata=float(initial_channel),
            ydata=0.0,
        )
    )

    figure.suptitle(
        f"Exact convolution validation viewer\n{resolved_path.name}",
        fontsize=14,
        fontweight="semibold",
    )

    if show:
        plt.show()

    return figure, {
        "heatmap": ax_heatmap,
        "signals": ax_signals,
        "error": ax_error,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: py -3 Portable/scripts/show_exact_convolution_validation.py <validation.csv>")
        return 1

    csv_argument = Path(sys.argv[1]).expanduser()
    if not csv_argument.exists():
        print(f"File not found: {csv_argument}")
        return 1

    try:
        plot_exact_convolution_validation(csv_argument, show=True)
    except ValueError as ex:
        print(ex)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

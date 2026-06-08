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


HEADER_RE = re.compile(r"^(Re|Im) H input (\d+) <- output (\d+)$")


def parse_frequency_matrix_csv(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    resolved_path = Path(csv_path).expanduser()
    with resolved_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, None)
        if not header:
            raise ValueError("CSV is empty.")
        if len(header) < 3 or (len(header) - 1) % 2 != 0:
            raise ValueError("Expected frequency column followed by Re/Im column pairs.")

        pairs: list[tuple[int, int, int, int]] = []
        max_input = -1
        max_output = -1
        for column in range(1, len(header), 2):
            if column + 1 >= len(header):
                raise ValueError("Missing imaginary column for the last response.")

            match_re = HEADER_RE.match(header[column].strip())
            match_im = HEADER_RE.match(header[column + 1].strip())
            if not match_re or not match_im:
                raise ValueError("Unexpected response header format.")
            if match_re.group(1) != "Re" or match_im.group(1) != "Im":
                raise ValueError("Expected Re/Im response pairs.")

            input_index = int(match_re.group(2))
            output_index = int(match_re.group(3))
            if (int(match_im.group(2)), int(match_im.group(3))) != (
                input_index,
                output_index,
            ):
                raise ValueError(
                    "Real and imaginary columns do not refer to the same channel pair."
                )

            pairs.append((input_index, output_index, column, column + 1))
            max_input = max(max_input, input_index)
            max_output = max(max_output, output_index)

        if max_input < 0 or max_output < 0:
            raise ValueError("No channel pairs were found in the CSV.")

        frequencies: list[float] = []
        re_columns = {
            (input_index, output_index): []
            for input_index, output_index, _, _ in pairs
        }
        im_columns = {
            (input_index, output_index): []
            for input_index, output_index, _, _ in pairs
        }

        for row in reader:
            if not row or not row[0].strip():
                continue

            frequencies.append(float(row[0]))
            for input_index, output_index, re_column, im_column in pairs:
                real_value = (
                    float(row[re_column])
                    if re_column < len(row) and row[re_column].strip()
                    else np.nan
                )
                imag_value = (
                    float(row[im_column])
                    if im_column < len(row) and row[im_column].strip()
                    else np.nan
                )
                re_columns[(input_index, output_index)].append(real_value)
                im_columns[(input_index, output_index)].append(imag_value)

    frequency_axis = np.asarray(frequencies, dtype=float)
    response_cube = np.full(
        (max_input + 1, max_output + 1, frequency_axis.size),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )

    for input_index, output_index, _, _ in pairs:
        real_part = np.asarray(re_columns[(input_index, output_index)], dtype=float)
        imag_part = np.asarray(im_columns[(input_index, output_index)], dtype=float)
        response_cube[input_index, output_index, :] = real_part + 1j * imag_part

    input_labels = [f"Input {index}" for index in range(max_input + 1)]
    output_labels = [f"Output {index}" for index in range(max_output + 1)]
    return frequency_axis, response_cube, input_labels, output_labels


def finite_complex_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values.real) & np.isfinite(values.imag)


def compute_summary_matrix(response_cube: np.ndarray) -> np.ndarray:
    magnitude = np.abs(response_cube)
    finite_mask = np.isfinite(magnitude)
    finite_counts = np.sum(finite_mask, axis=2)
    squared_magnitude = np.where(finite_mask, np.square(magnitude), 0.0)
    mean_squared_magnitude = np.divide(
        np.sum(squared_magnitude, axis=2),
        finite_counts,
        out=np.full(finite_counts.shape, np.nan, dtype=float),
        where=finite_counts > 0,
    )
    broadband_rms = np.sqrt(mean_squared_magnitude)
    return 20.0 * np.log10(np.maximum(broadband_rms, 1.0e-12))


def strongest_finite_pairs(summary_db: np.ndarray, count: int = 8) -> np.ndarray:
    finite_summary = np.isfinite(summary_db)
    finite_count = int(np.count_nonzero(finite_summary))
    if finite_count == 0:
        return np.empty((0, 2), dtype=int)

    strongest_count = min(count, finite_count)
    strongest_flat = np.argsort(
        np.where(finite_summary, summary_db, -np.inf),
        axis=None,
    )[::-1][:strongest_count]
    return np.column_stack(np.unravel_index(strongest_flat, summary_db.shape))


def _draw_selected_pair(
    figure,
    ax_heatmap,
    ax_magnitude,
    ax_phase,
    ax_impulse,
    highlight,
    *,
    input_index: int,
    output_index: int,
    response_cube: np.ndarray,
    frequency_axis: np.ndarray,
    valid_frequency_mask: np.ndarray,
    impulse_sample_count: int,
    impulse_time_ms: np.ndarray,
    input_labels: list[str],
    output_labels: list[str],
    summary_db: np.ndarray,
) -> None:
    selected_response = response_cube[input_index, output_index, :]
    selected_finite_mask = finite_complex_mask(selected_response)
    selected_valid_mask = valid_frequency_mask & selected_finite_mask
    selected_response_for_impulse = np.where(
        selected_finite_mask,
        selected_response,
        0.0 + 0.0j,
    )
    selected_impulse = np.fft.irfft(
        selected_response_for_impulse,
        n=impulse_sample_count,
    )

    highlight.set_xy((output_index - 0.5, input_index - 0.5))

    ax_magnitude.clear()
    ax_phase.clear()
    ax_impulse.clear()

    ax_magnitude.set_ylabel("Magnitude (dB)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_impulse.set_ylabel("Impulse")
    ax_impulse.set_xlabel("Time (ms)")
    ax_impulse.set_title("Impulse response")

    if not np.any(selected_valid_mask):
        ax_magnitude.set_title(
            f"{input_labels[input_index]} <- {output_labels[output_index]} | "
            "no valid frequency data"
        )
        ax_magnitude.text(
            0.5,
            0.5,
            "This channel pair has no finite frequency-response samples.",
            transform=ax_magnitude.transAxes,
            va="center",
            ha="center",
            bbox={
                "facecolor": "#fffaf2",
                "edgecolor": "#d6c7b3",
                "boxstyle": "round,pad=0.4",
            },
        )
        ax_phase.text(
            0.5,
            0.5,
            "Phase unavailable",
            transform=ax_phase.transAxes,
            va="center",
            ha="center",
        )
        ax_impulse.text(
            0.5,
            0.5,
            "Impulse unavailable",
            transform=ax_impulse.transAxes,
            va="center",
            ha="center",
        )
        figure.canvas.draw_idle()
        return

    selected_frequency = frequency_axis[selected_valid_mask]
    selected_magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(selected_response[selected_valid_mask]), 1.0e-12)
    )
    selected_phase_deg = np.rad2deg(
        np.unwrap(np.angle(selected_response[selected_valid_mask]))
    )

    ax_magnitude.semilogx(
        selected_frequency,
        selected_magnitude_db,
        color="#0f766e",
        linewidth=2.1,
    )
    ax_phase.semilogx(
        selected_frequency,
        selected_phase_deg,
        color="#c2410c",
        linewidth=1.9,
    )
    ax_impulse.plot(
        impulse_time_ms,
        selected_impulse,
        color="#1d4ed8",
        linewidth=1.8,
    )

    peak_index = int(np.nanargmax(selected_magnitude_db))
    impulse_peak_index = int(np.nanargmax(np.abs(selected_impulse)))
    ax_magnitude.scatter(
        [selected_frequency[peak_index]],
        [selected_magnitude_db[peak_index]],
        color="#f97316",
        s=36,
        zorder=3,
    )
    ax_impulse.scatter(
        [impulse_time_ms[impulse_peak_index]],
        [selected_impulse[impulse_peak_index]],
        color="#f97316",
        s=36,
        zorder=3,
    )

    ax_magnitude.grid(True, which="both", alpha=0.22)
    ax_phase.grid(True, which="both", alpha=0.22)
    ax_impulse.grid(True, alpha=0.22)
    ax_magnitude.set_title(
        f"{input_labels[input_index]} <- {output_labels[output_index]} | "
        f"broadband {summary_db[input_index, output_index]:.1f} dB"
    )
    ax_magnitude.text(
        0.02,
        0.96,
        "\n".join(
            [
                f"Peak: {selected_magnitude_db[peak_index]:.1f} dB",
                f"At: {selected_frequency[peak_index]:.1f} Hz",
                f"Median: {np.nanmedian(selected_magnitude_db):.1f} dB",
            ]
        ),
        transform=ax_magnitude.transAxes,
        va="top",
        ha="left",
        bbox={
            "facecolor": "#fffaf2",
            "edgecolor": "#d6c7b3",
            "boxstyle": "round,pad=0.4",
        },
    )
    ax_impulse.text(
        0.02,
        0.96,
        "\n".join(
            [
                f"Peak: {selected_impulse[impulse_peak_index]:.3g}",
                f"At: {impulse_time_ms[impulse_peak_index]:.2f} ms",
                f"RMS: {np.sqrt(np.mean(np.square(selected_impulse))):.3g}",
            ]
        ),
        transform=ax_impulse.transAxes,
        va="top",
        ha="left",
        bbox={
            "facecolor": "#fffaf2",
            "edgecolor": "#d6c7b3",
            "boxstyle": "round,pad=0.4",
        },
    )
    figure.canvas.draw_idle()


def plot_topology_matrix(
    csv_path: str | Path,
    *,
    show: bool = True,
    selected_pair: tuple[int, int] | None = None,
):
    resolved_path = Path(csv_path).expanduser()
    frequency_axis, response_cube, input_labels, output_labels = (
        parse_frequency_matrix_csv(resolved_path)
    )
    summary_db = compute_summary_matrix(response_cube)

    valid_pair_mask = np.any(finite_complex_mask(response_cube), axis=2)
    invalid_pairs = np.column_stack(np.where(~valid_pair_mask))
    if invalid_pairs.size > 0:
        preview_pairs = ", ".join(
            f"in {input_index} <- out {output_index}"
            for input_index, output_index in invalid_pairs[:8]
        )
        suffix = " ..." if len(invalid_pairs) > 8 else ""
        print(
            "Warning: "
            f"{len(invalid_pairs)} channel pair(s) contain no finite "
            f"frequency-response data: {preview_pairs}{suffix}"
        )

    if not np.any(np.isfinite(summary_db)):
        raise ValueError(
            "The frequency-domain matrix contains no finite channel-pair summary values."
        )

    impulse_sample_count = max(1, 2 * (frequency_axis.size - 1))
    if frequency_axis.size > 1 and frequency_axis[1] > 0.0:
        sample_rate_estimate = frequency_axis[1] * impulse_sample_count
    else:
        sample_rate_estimate = float(impulse_sample_count)
    impulse_time_ms = (
        1000.0 * np.arange(impulse_sample_count, dtype=float)
        / max(sample_rate_estimate, 1.0)
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

    figure = plt.figure(figsize=(14.5, 9.2), constrained_layout=True)
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=[1.05, 1.25],
        height_ratios=[1.0, 1.0, 0.95],
    )
    ax_heatmap = figure.add_subplot(grid[:, 0])
    ax_magnitude = figure.add_subplot(grid[0, 1])
    ax_phase = figure.add_subplot(grid[1, 1], sharex=ax_magnitude)
    ax_impulse = figure.add_subplot(grid[2, 1])

    heatmap = ax_heatmap.imshow(
        summary_db,
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    colorbar = figure.colorbar(heatmap, ax=ax_heatmap, pad=0.03, shrink=0.88)
    colorbar.set_label("Broadband magnitude (dB)")

    input_step = max(1, len(input_labels) // 8)
    output_step = max(1, len(output_labels) // 8)
    ax_heatmap.set_xticks(np.arange(0, len(output_labels), output_step))
    ax_heatmap.set_yticks(np.arange(0, len(input_labels), input_step))
    ax_heatmap.set_xticklabels(
        [str(index) for index in range(0, len(output_labels), output_step)]
    )
    ax_heatmap.set_yticklabels(
        [str(index) for index in range(0, len(input_labels), input_step)]
    )
    ax_heatmap.set_xlabel("Output channel")
    ax_heatmap.set_ylabel("Input channel")
    ax_heatmap.set_title("Topology heatmap\n(click a cell to inspect frequency response)")
    ax_heatmap.set_xticks(np.arange(-0.5, len(output_labels), 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, len(input_labels), 1), minor=True)
    ax_heatmap.grid(which="minor", color="#ffffff", linewidth=0.35, alpha=0.18)
    ax_heatmap.tick_params(which="minor", bottom=False, left=False)

    highlight = Rectangle(
        (-0.5, -0.5),
        1.0,
        1.0,
        fill=False,
        linewidth=2.4,
        edgecolor="#f97316",
    )
    ax_heatmap.add_patch(highlight)

    strongest_pairs = strongest_finite_pairs(summary_db)
    if strongest_pairs.size > 0:
        ax_heatmap.scatter(
            strongest_pairs[:, 1],
            strongest_pairs[:, 0],
            s=26,
            facecolors="none",
            edgecolors="#f8fafc",
            linewidths=0.9,
        )
        strongest_lines = [
            f"{rank + 1}. in {input_index} <- out {output_index}: "
            f"{summary_db[input_index, output_index]:.1f} dB"
            for rank, (input_index, output_index) in enumerate(strongest_pairs)
        ]
        ax_heatmap.text(
            1.03,
            0.02,
            "Strongest pairs\n" + "\n".join(strongest_lines),
            transform=ax_heatmap.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            bbox={
                "facecolor": "#fffaf2",
                "edgecolor": "#d6c7b3",
                "boxstyle": "round,pad=0.45",
            },
        )

    ax_magnitude.set_ylabel("Magnitude (dB)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_impulse.set_ylabel("Impulse")
    ax_impulse.set_xlabel("Time (ms)")

    valid_frequency_mask = frequency_axis > 0.0
    if not np.any(valid_frequency_mask):
        valid_frequency_mask = np.ones_like(frequency_axis, dtype=bool)

    def on_click(event) -> None:
        if event.inaxes is not ax_heatmap or event.xdata is None or event.ydata is None:
            return

        output_index = int(round(event.xdata))
        input_index = int(round(event.ydata))
        if not (
            0 <= input_index < response_cube.shape[0]
            and 0 <= output_index < response_cube.shape[1]
        ):
            return

        _draw_selected_pair(
            figure,
            ax_heatmap,
            ax_magnitude,
            ax_phase,
            ax_impulse,
            highlight,
            input_index=input_index,
            output_index=output_index,
            response_cube=response_cube,
            frequency_axis=frequency_axis,
            valid_frequency_mask=valid_frequency_mask,
            impulse_sample_count=impulse_sample_count,
            impulse_time_ms=impulse_time_ms,
            input_labels=input_labels,
            output_labels=output_labels,
            summary_db=summary_db,
        )

    figure.canvas.mpl_connect("button_press_event", on_click)

    if selected_pair is None:
        initial_input, initial_output = strongest_pairs[0]
    else:
        initial_input, initial_output = selected_pair
    on_click(
        SimpleNamespace(
            inaxes=ax_heatmap,
            xdata=float(initial_output),
            ydata=float(initial_input),
        )
    )

    figure.suptitle(
        f"Frequency-domain topology viewer\n{resolved_path.name}",
        fontsize=14,
        fontweight="semibold",
    )

    if show:
        plt.show()

    return figure, {
        "heatmap": ax_heatmap,
        "magnitude": ax_magnitude,
        "phase": ax_phase,
        "impulse": ax_impulse,
    }


def display_topology_matrix_widget(csv_path: str | Path):
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError:
        return plot_topology_matrix(csv_path, show=True)

    resolved_path = Path(csv_path).expanduser()
    frequency_axis, response_cube, input_labels, output_labels = (
        parse_frequency_matrix_csv(resolved_path)
    )
    summary_db = compute_summary_matrix(response_cube)
    strongest_pairs_array = strongest_finite_pairs(summary_db)
    strongest_pairs = [
        (int(input_index), int(output_index))
        for input_index, output_index in strongest_pairs_array
    ]

    input_slider = widgets.IntSlider(
        value=strongest_pairs[0][0] if strongest_pairs else 0,
        min=0,
        max=max(0, response_cube.shape[0] - 1),
        step=1,
        description="Input",
        continuous_update=False,
    )
    output_slider = widgets.IntSlider(
        value=strongest_pairs[0][1] if strongest_pairs else 0,
        min=0,
        max=max(0, response_cube.shape[1] - 1),
        step=1,
        description="Output",
        continuous_update=False,
    )
    hotspot_dropdown = widgets.Dropdown(
        options=[
            (
                f"in {input_index} <- out {output_index} "
                f"({summary_db[input_index, output_index]:.1f} dB)",
                (input_index, output_index),
            )
            for input_index, output_index in strongest_pairs
        ],
        description="Hotspot",
    )
    hint = widgets.HTML(
        value=(
            "<b>Selection fallback:</b> this notebook is using a static "
            "Matplotlib backend, so direct clicking on the heatmap is "
            "unavailable here. Use the controls below."
        )
    )
    output = widgets.Output()

    def render_selected_pair(input_index: int, output_index: int) -> None:
        with output:
            clear_output(wait=True)
            figure, axes = plot_topology_matrix(
                resolved_path,
                show=False,
                selected_pair=(input_index, output_index),
            )
            display(figure)
            plt.close(figure)

    def sync_from_sliders(_change=None) -> None:
        selected_pair = (int(input_slider.value), int(output_slider.value))
        if hotspot_dropdown.options:
            for _, value in hotspot_dropdown.options:
                if value == selected_pair:
                    hotspot_dropdown.value = value
                    break
        render_selected_pair(*selected_pair)

    def sync_from_dropdown(change) -> None:
        if change["name"] != "value" or change["new"] is None:
            return
        input_index, output_index = change["new"]
        input_slider.value = int(input_index)
        output_slider.value = int(output_index)

    input_slider.observe(sync_from_sliders, names="value")
    output_slider.observe(sync_from_sliders, names="value")
    hotspot_dropdown.observe(sync_from_dropdown, names="value")

    controls = widgets.VBox(
        [
            hint,
            widgets.HBox([input_slider, output_slider]),
            hotspot_dropdown,
            output,
        ]
    )
    display(controls)
    render_selected_pair(int(input_slider.value), int(output_slider.value))
    return controls


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: py -3 Portable/scripts/show_topology_matrix.py <frequency_matrix.csv>")
        return 1

    csv_argument = Path(sys.argv[1]).expanduser()
    if not csv_argument.exists():
        print(f"File not found: {csv_argument}")
        return 1

    try:
        plot_topology_matrix(csv_argument, show=True)
    except ValueError as ex:
        print(ex)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

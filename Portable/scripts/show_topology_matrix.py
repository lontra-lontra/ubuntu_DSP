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
TIME_HEADER_RE = re.compile(r"^h input (\d+) <- output (\d+)$")
PLAYED_HEADER_RE = re.compile(r"^played ch (\d+)$")
RECORDED_HEADER_RE = re.compile(r"^recorded ch (\d+)$")


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


def infer_capture_base_path(csv_path: str | Path) -> Path:
    resolved_path = Path(csv_path).expanduser()
    name = resolved_path.name
    for suffix in ("_frequency_domain_matrix.csv", "_time_domain_matrix.csv"):
        if name.endswith(suffix):
            return resolved_path.with_name(name[: -len(suffix)])
    return resolved_path.with_suffix("")


def parse_time_matrix_csv(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_path = Path(csv_path).expanduser()
    with resolved_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, None)
        if not header:
            raise ValueError("Time-domain CSV is empty.")
        if len(header) < 2:
            raise ValueError("Time-domain CSV must contain a time column and at least one response.")

        pairs: list[tuple[int, int, int]] = []
        max_input = -1
        max_output = -1
        for column in range(1, len(header)):
            match = TIME_HEADER_RE.match(header[column].strip())
            if not match:
                raise ValueError("Unexpected time-domain response header format.")

            input_index = int(match.group(1))
            output_index = int(match.group(2))
            pairs.append((input_index, output_index, column))
            max_input = max(max_input, input_index)
            max_output = max(max_output, output_index)

        time_axis_ms: list[float] = []
        response_columns = {
            (input_index, output_index): []
            for input_index, output_index, _ in pairs
        }

        for row in reader:
            if not row or not row[0].strip():
                continue

            time_axis_ms.append(1000.0 * float(row[0]))
            for input_index, output_index, column in pairs:
                value = (
                    float(row[column])
                    if column < len(row) and row[column].strip()
                    else np.nan
                )
                response_columns[(input_index, output_index)].append(value)

    impulse_cube = np.full(
        (max_input + 1, max_output + 1, len(time_axis_ms)),
        np.nan,
        dtype=float,
    )
    for input_index, output_index, _ in pairs:
        impulse_cube[input_index, output_index, :] = np.asarray(
            response_columns[(input_index, output_index)],
            dtype=float,
        )

    return np.asarray(time_axis_ms, dtype=float), impulse_cube


def parse_raw_capture_csv(
    csv_path: str | Path,
) -> SimpleNamespace:
    resolved_path = Path(csv_path).expanduser()
    with resolved_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, None)
        if not header:
            raise ValueError("Raw capture CSV is empty.")
        if len(header) < 3:
            raise ValueError("Raw capture CSV must contain time plus played/recorded channels.")

        played_columns: dict[int, int] = {}
        recorded_columns: dict[int, int] = {}
        max_channel = -1
        for column, name in enumerate(header[1:], start=1):
            played_match = PLAYED_HEADER_RE.match(name.strip())
            recorded_match = RECORDED_HEADER_RE.match(name.strip())
            if played_match:
                channel = int(played_match.group(1))
                played_columns[channel] = column
                max_channel = max(max_channel, channel)
            elif recorded_match:
                channel = int(recorded_match.group(1))
                recorded_columns[channel] = column
                max_channel = max(max_channel, channel)

        if max_channel < 0:
            raise ValueError("Raw capture CSV contains no played/recorded channel columns.")

        time_axis_ms: list[float] = []
        played_rows: list[list[float]] = []
        recorded_rows: list[list[float]] = []

        for row in reader:
            if not row or not row[0].strip():
                continue

            padded = row + [""] * (len(header) - len(row))
            time_axis_ms.append(1000.0 * float(padded[0]))

            played_sample_row = [np.nan] * (max_channel + 1)
            recorded_sample_row = [np.nan] * (max_channel + 1)
            for channel, column in played_columns.items():
                played_sample_row[channel] = (
                    float(padded[column]) if padded[column].strip() else np.nan
                )
            for channel, column in recorded_columns.items():
                recorded_sample_row[channel] = (
                    float(padded[column]) if padded[column].strip() else np.nan
                )
            played_rows.append(played_sample_row)
            recorded_rows.append(recorded_sample_row)

    played_matrix = np.asarray(played_rows, dtype=float).T
    recorded_matrix = np.asarray(recorded_rows, dtype=float).T
    return SimpleNamespace(
        time_ms=np.asarray(time_axis_ms, dtype=float),
        played=played_matrix,
        recorded=recorded_matrix,
        path=resolved_path,
    )


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


def compute_raw_input_peak_matrix(
    output_count: int,
    input_count: int,
    raw_capture_loader,
) -> np.ndarray:
    peak_matrix = np.full((input_count, output_count), np.nan, dtype=float)

    for output_index in range(output_count):
        raw_capture = raw_capture_loader(output_index)
        if raw_capture is None:
            continue

        available_input_count = min(input_count, raw_capture.recorded.shape[0])
        for input_index in range(available_input_count):
            recorded_signal = raw_capture.recorded[input_index, :]
            finite_mask = np.isfinite(recorded_signal)
            if not np.any(finite_mask):
                continue
            peak_matrix[input_index, output_index] = np.nanmax(
                np.abs(recorded_signal[finite_mask])
            )

    return peak_matrix


def _draw_selected_pair(
    figure,
    ax_heatmap,
    ax_output_signal,
    ax_input_signal,
    ax_impulse,
    highlight,
    *,
    input_index: int,
    output_index: int,
    response_cube: np.ndarray,
    fallback_impulse_time_ms: np.ndarray,
    input_labels: list[str],
    output_labels: list[str],
    raw_peak_matrix: np.ndarray,
    impulse_cube: np.ndarray | None,
    impulse_time_ms: np.ndarray | None,
    raw_capture_loader,
) -> None:
    highlight.set_xy((output_index - 0.5, input_index - 0.5))

    ax_output_signal.clear()
    ax_input_signal.clear()
    ax_impulse.clear()

    ax_output_signal.set_ylabel("Played output")
    ax_output_signal.set_xlabel("Time (ms)")
    ax_input_signal.set_ylabel("Recorded input")
    ax_input_signal.set_xlabel("Time (ms)")
    ax_impulse.set_ylabel("Impulse")
    ax_impulse.set_xlabel("Time (ms)")
    ax_output_signal.set_title(f"{output_labels[output_index]} played signal")
    raw_peak_text = ""
    if (
        input_index < raw_peak_matrix.shape[0]
        and output_index < raw_peak_matrix.shape[1]
        and np.isfinite(raw_peak_matrix[input_index, output_index])
    ):
        raw_peak_text = f" | peak {raw_peak_matrix[input_index, output_index]:.3g}"
    ax_input_signal.set_title(
        f"{input_labels[input_index]} recorded signal{raw_peak_text}"
    )
    ax_impulse.set_title(f"Impulse response for {input_labels[input_index]} <- {output_labels[output_index]}")

    raw_capture = raw_capture_loader(output_index)
    if raw_capture is None:
        for axis, message in (
            (ax_output_signal, "Raw capture for this output channel is unavailable."),
            (ax_input_signal, "Recorded input trace is unavailable."),
        ):
            axis.text(
                0.5,
                0.5,
                message,
                transform=axis.transAxes,
                va="center",
                ha="center",
                bbox={
                    "facecolor": "#fffaf2",
                    "edgecolor": "#d6c7b3",
                    "boxstyle": "round,pad=0.4",
                },
            )
    else:
        played_signal = raw_capture.played[output_index, :]
        recorded_signal = raw_capture.recorded[input_index, :]
        time_axis_ms = raw_capture.time_ms

        ax_output_signal.plot(
            time_axis_ms,
            played_signal,
            color="#0f766e",
            linewidth=1.8,
            label=f"{output_labels[output_index]} played",
        )
        ax_input_signal.plot(
            time_axis_ms,
            recorded_signal,
            color="#c2410c",
            linewidth=1.3,
            label=f"{input_labels[input_index]} recorded",
        )
        x_min = float(time_axis_ms[0]) if time_axis_ms.size else 0.0
        x_max = float(time_axis_ms[-1]) if time_axis_ms.size else 0.0
        ax_output_signal.set_xlim(x_min, x_max)
        ax_input_signal.set_xlim(x_min, x_max)
        ax_output_signal.grid(True, alpha=0.22)
        ax_input_signal.grid(True, alpha=0.22)
        ax_output_signal.legend(loc="upper right")
        ax_input_signal.legend(loc="upper right")
        ax_output_signal.text(
            0.02,
            0.96,
            "\n".join(
                [
                    f"Peak: {np.nanmax(np.abs(played_signal)):.3g}",
                    f"RMS: {np.sqrt(np.nanmean(np.square(played_signal))):.3g}",
                    f"Time window: {x_min:.1f} to {x_max:.1f} ms",
                ]
            ),
            transform=ax_output_signal.transAxes,
            va="top",
            ha="left",
            bbox={
                "facecolor": "#fffaf2",
                "edgecolor": "#d6c7b3",
                "boxstyle": "round,pad=0.4",
            },
        )
        ax_input_signal.text(
            0.02,
            0.96,
            "\n".join(
                [
                    f"Peak: {np.nanmax(np.abs(recorded_signal)):.3g}",
                    f"RMS: {np.sqrt(np.nanmean(np.square(recorded_signal))):.3g}",
                    f"Output channel: {output_index}",
                ]
            ),
            transform=ax_input_signal.transAxes,
            va="top",
            ha="left",
            bbox={
                "facecolor": "#fffaf2",
                "edgecolor": "#d6c7b3",
                "boxstyle": "round,pad=0.4",
            },
        )

    selected_impulse = None
    selected_impulse_time_ms = None
    if (
        impulse_cube is not None
        and impulse_time_ms is not None
        and input_index < impulse_cube.shape[0]
        and output_index < impulse_cube.shape[1]
    ):
        candidate_impulse = impulse_cube[input_index, output_index, :]
        if np.any(np.isfinite(candidate_impulse)):
            selected_impulse = candidate_impulse
            selected_impulse_time_ms = impulse_time_ms

    if selected_impulse is None:
        selected_response = response_cube[input_index, output_index, :]
        selected_finite_mask = finite_complex_mask(selected_response)
        if np.any(selected_finite_mask):
            selected_response_for_impulse = np.where(
                selected_finite_mask,
                selected_response,
                0.0 + 0.0j,
            )
            selected_impulse = np.fft.irfft(
                selected_response_for_impulse,
                n=fallback_impulse_time_ms.size,
            )
            selected_impulse_time_ms = fallback_impulse_time_ms

    if selected_impulse is None or selected_impulse_time_ms is None:
        ax_impulse.text(
            0.5,
            0.5,
            "Impulse unavailable",
            transform=ax_impulse.transAxes,
            va="center",
            ha="center",
            bbox={
                "facecolor": "#fffaf2",
                "edgecolor": "#d6c7b3",
                "boxstyle": "round,pad=0.4",
            },
        )
        figure.canvas.draw_idle()
        return

    selected_impulse = np.asarray(selected_impulse, dtype=float)
    selected_impulse_time_ms = np.asarray(selected_impulse_time_ms, dtype=float)
    finite_impulse_mask = np.isfinite(selected_impulse)
    if not np.any(finite_impulse_mask):
        ax_impulse.text(
            0.5,
            0.5,
            "Impulse unavailable",
            transform=ax_impulse.transAxes,
            va="center",
            ha="center",
            bbox={
                "facecolor": "#fffaf2",
                "edgecolor": "#d6c7b3",
                "boxstyle": "round,pad=0.4",
            },
        )
        figure.canvas.draw_idle()
        return

    ax_impulse.plot(
        selected_impulse_time_ms,
        selected_impulse,
        color="#1d4ed8",
        linewidth=1.8,
    )
    impulse_peak_index = int(np.nanargmax(np.abs(selected_impulse)))
    ax_impulse.scatter(
        [selected_impulse_time_ms[impulse_peak_index]],
        [selected_impulse[impulse_peak_index]],
        color="#f97316",
        s=36,
        zorder=3,
    )
    ax_impulse.grid(True, alpha=0.22)
    ax_impulse.text(
        0.02,
        0.96,
        "\n".join(
            [
                f"Peak: {selected_impulse[impulse_peak_index]:.3g}",
                f"At: {selected_impulse_time_ms[impulse_peak_index]:.2f} ms",
                f"RMS: {np.sqrt(np.nanmean(np.square(selected_impulse))):.3g}",
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
    frequency_summary_db = compute_summary_matrix(response_cube)

    impulse_sample_count = max(1, 2 * (frequency_axis.size - 1))
    if frequency_axis.size > 1 and frequency_axis[1] > 0.0:
        sample_rate_estimate = frequency_axis[1] * impulse_sample_count
    else:
        sample_rate_estimate = float(impulse_sample_count)
    fallback_impulse_time_ms = (
        1000.0 * np.arange(impulse_sample_count, dtype=float)
        / max(sample_rate_estimate, 1.0)
    )

    base_path = infer_capture_base_path(resolved_path)
    time_matrix_path = Path(f"{base_path}_time_domain_matrix.csv")
    impulse_time_ms = None
    impulse_cube = None
    if time_matrix_path.exists() and time_matrix_path.stat().st_size > 0:
        try:
            impulse_time_ms, impulse_cube = parse_time_matrix_csv(time_matrix_path)
        except ValueError as ex:
            print(f"Warning: could not load time-domain matrix {time_matrix_path.name}: {ex}")

    raw_capture_cache: dict[int, SimpleNamespace | None] = {}

    def load_raw_capture_for_output(output_index: int) -> SimpleNamespace | None:
        if output_index in raw_capture_cache:
            return raw_capture_cache[output_index]

        raw_csv_path = Path(f"{base_path}_raw_capture_output_{output_index}.csv")
        if not raw_csv_path.exists() or raw_csv_path.stat().st_size == 0:
            raw_capture_cache[output_index] = None
            return None

        try:
            raw_capture_cache[output_index] = parse_raw_capture_csv(raw_csv_path)
        except ValueError as ex:
            print(f"Warning: could not load raw capture {raw_csv_path.name}: {ex}")
            raw_capture_cache[output_index] = None
        return raw_capture_cache[output_index]

    raw_peak_matrix = compute_raw_input_peak_matrix(
        response_cube.shape[1],
        response_cube.shape[0],
        load_raw_capture_for_output,
    )
    selector_matrix = raw_peak_matrix
    selector_label = "Recorded input peak"

    if not np.any(np.isfinite(selector_matrix)):
        selector_matrix = frequency_summary_db
        selector_label = "Frequency-response summary (dB)"
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

    if not np.any(np.isfinite(selector_matrix)):
        raise ValueError(
            "The topology viewer could not find any finite raw captures or frequency-response summaries."
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

    figure = plt.figure(figsize=(14.5, 10.2), constrained_layout=True)
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=[1.05, 1.25],
        height_ratios=[1.0, 1.0, 1.0],
    )
    ax_heatmap = figure.add_subplot(grid[:, 0])
    ax_output_signal = figure.add_subplot(grid[0, 1])
    ax_input_signal = figure.add_subplot(grid[1, 1], sharex=ax_output_signal)
    ax_impulse = figure.add_subplot(grid[2, 1])

    heatmap = ax_heatmap.imshow(
        selector_matrix,
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    colorbar = figure.colorbar(heatmap, ax=ax_heatmap, pad=0.03, shrink=0.88)
    colorbar.set_label(selector_label)

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
    ax_heatmap.set_title("Pair selector\n(click a cell to inspect one channel pair)")
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

    strongest_pairs = strongest_finite_pairs(selector_matrix)
    if strongest_pairs.size > 0:
        ax_heatmap.scatter(
            strongest_pairs[:, 1],
            strongest_pairs[:, 0],
            s=26,
            facecolors="none",
            edgecolors="#f8fafc",
            linewidths=0.9,
        )

    ax_output_signal.set_ylabel("Played output")
    ax_output_signal.set_xlabel("Time (ms)")
    ax_input_signal.set_ylabel("Recorded input")
    ax_input_signal.set_xlabel("Time (ms)")
    ax_impulse.set_ylabel("Impulse")
    ax_impulse.set_xlabel("Time (ms)")

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
            ax_output_signal,
            ax_input_signal,
            ax_impulse,
            highlight,
            input_index=input_index,
            output_index=output_index,
            response_cube=response_cube,
            fallback_impulse_time_ms=fallback_impulse_time_ms,
            input_labels=input_labels,
            output_labels=output_labels,
            raw_peak_matrix=raw_peak_matrix,
            impulse_cube=impulse_cube,
            impulse_time_ms=impulse_time_ms,
            raw_capture_loader=load_raw_capture_for_output,
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
        f"Topology viewer\n{resolved_path.name}",
        fontsize=14,
        fontweight="semibold",
    )

    if show:
        plt.show()

    return figure, {
        "heatmap": ax_heatmap,
        "output_signal": ax_output_signal,
        "input_signal": ax_input_signal,
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
    base_path = infer_capture_base_path(resolved_path)
    raw_capture_cache: dict[int, SimpleNamespace | None] = {}

    def load_raw_capture_for_output(output_index: int) -> SimpleNamespace | None:
        if output_index in raw_capture_cache:
            return raw_capture_cache[output_index]

        raw_csv_path = Path(f"{base_path}_raw_capture_output_{output_index}.csv")
        if not raw_csv_path.exists() or raw_csv_path.stat().st_size == 0:
            raw_capture_cache[output_index] = None
            return None

        try:
            raw_capture_cache[output_index] = parse_raw_capture_csv(raw_csv_path)
        except ValueError:
            raw_capture_cache[output_index] = None
        return raw_capture_cache[output_index]

    raw_peak_matrix = compute_raw_input_peak_matrix(
        response_cube.shape[1],
        response_cube.shape[0],
        load_raw_capture_for_output,
    )
    selector_matrix = raw_peak_matrix
    selector_label = "peak"
    if not np.any(np.isfinite(selector_matrix)):
        selector_matrix = compute_summary_matrix(response_cube)
        selector_label = "summary dB"

    strongest_pairs_array = strongest_finite_pairs(selector_matrix)
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
                f"({selector_matrix[input_index, output_index]:.3g} {selector_label})",
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

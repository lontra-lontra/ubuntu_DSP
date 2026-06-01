import csv
import math
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    print("This plotter needs matplotlib. Install it with: python3 -m pip install matplotlib")
    raise SystemExit(1)


RESPONSE_COLOR = "#0f766e"
SMALLEST_COLOR = "#2563eb"
MEDIAN_COLOR = "#ca8a04"
BIGGEST_COLOR = "#dc2626"
REST_COLOR = "#6b7280"


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


def values_for_field(rows: list[dict[str, float]], field: str) -> list[float]:
    return [row[field] for row in rows]


def series_match(lhs: list[float], rhs: list[float], tolerance: float = 1.0e-6) -> bool:
    if len(lhs) != len(rhs):
        return False

    for lhs_value, rhs_value in zip(lhs, rhs):
        if math.isnan(lhs_value) and math.isnan(rhs_value):
            continue
        if math.isnan(lhs_value) != math.isnan(rhs_value):
            return False
        if abs(lhs_value - rhs_value) > tolerance:
            return False

    return True


def infer_delay_index(signal: list[float], threshold_fraction: float = 0.5) -> int | None:
    peak = 0.0
    for value in signal:
        if math.isnan(value):
            continue
        peak = max(peak, abs(value))

    if peak <= 0.0:
        return None

    threshold = peak * threshold_fraction
    for index, value in enumerate(signal):
        if not math.isnan(value) and abs(value) >= threshold:
            return index

    return None


def load_selected_delay_profiles(csv_path: Path) -> dict[str, list[float]]:
    selected_delays_path = csv_path.with_name(
        "superpose_signals_and_responses_selected_delays.csv"
    )
    if not selected_delays_path.exists():
        return {}

    fieldnames, rows = load_rows(selected_delays_path)
    required_fields = {
        "normalized_response_smallest_delay",
        "normalized_response_median_delay",
        "normalized_response_biggest_delay",
    }
    if not required_fields.issubset(fieldnames) or not rows:
        return {}

    return {
        "smallest": values_for_field(rows, "normalized_response_smallest_delay"),
        "median": values_for_field(rows, "normalized_response_median_delay"),
        "biggest": values_for_field(rows, "normalized_response_biggest_delay"),
    }


def attach_toggle_legend(ax, groups: list[dict[str, object]]) -> None:
    visible_groups = [group for group in groups if group["lines"]]
    if not visible_groups:
        return

    handles = [
        Line2D(
            [0],
            [0],
            color=group["color"],
            linestyle=group.get("linestyle", "-"),
            linewidth=group.get("linewidth", 1.8),
        )
        for group in visible_groups
    ]
    labels = [str(group["label"]) for group in visible_groups]
    legend = ax.legend(handles=handles, labels=labels)

    linked: dict[Line2D, list[object]] = {}
    for legend_line, group in zip(legend.get_lines(), visible_groups):
        legend_line.set_picker(True)
        linked[legend_line] = list(group["lines"])

    def refresh_legend_alpha() -> None:
        for legend_line, lines in linked.items():
            visible = any(line.get_visible() for line in lines)
            legend_line.set_alpha(1.0 if visible else 0.2)

    def on_pick(event) -> None:
        legend_line = event.artist
        if legend_line not in linked:
            return

        target_lines = linked[legend_line]
        visible = not any(line.get_visible() for line in target_lines)
        for line in target_lines:
            line.set_visible(visible)

        refresh_legend_alpha()
        ax.figure.canvas.draw_idle()

    refresh_legend_alpha()
    ax.figure.canvas.mpl_connect("pick_event", on_pick)


def parse_arguments(argv: list[str]) -> Path | None:
    if len(argv) == 2:
        return Path(argv[1]).expanduser()

    if len(argv) == 3 and argv[1] in {"--show", "--save"}:
        return Path(argv[2]).expanduser()

    print(
        "Usage: python3 Portable/scripts/plot_superpose_signals_and_responses.py "
        "<file.csv>"
    )
    return None


def finalize_figure(figure, csv_path: Path) -> Path:
    output_path = csv_path.with_suffix(".png")
    figure.savefig(output_path, dpi=160)
    plt.show()
    plt.close(figure)
    return output_path


def build_response_colors(count: int) -> list[tuple[float, float, float, float]]:
    if count <= 0:
        return []

    cmap = plt.get_cmap("turbo")
    if count == 1:
        return [cmap(0.18)]

    return [
        cmap(0.08 + (0.84 * index / float(count - 1)))
        for index in range(count)
    ]


def sorted_response_fields(fieldnames: list[str]) -> list[str]:
    response_fields = [
        field for field in fieldnames if field.startswith("response_rep_")
    ]
    return sorted(
        response_fields,
        key=lambda field: int(field.rsplit("_", maxsplit=1)[-1]),
    )


def plot_single_response(csv_path: Path, rows: list[dict[str, float]]):
    time_ms = [row["time_ms"] for row in rows]
    response = [row["normalized_response"] for row in rows]
    delay_ms = rows[0].get("delay_ms_marker", float("nan")) if rows else float("nan")

    figure, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        time_ms,
        response,
        linewidth=1.8,
        label="normalized_response",
        color=RESPONSE_COLOR,
    )
    if not math.isnan(delay_ms) and delay_ms >= 0.0:
        ax.axvline(
            delay_ms,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"delay = {delay_ms:.3f} ms",
        )
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized amplitude")
    ax.set_title(csv_path.stem)
    ax.grid(True, alpha=0.3)
    ax.legend()
    figure.tight_layout()
    return figure


def plot_selected_delays(csv_path: Path, rows: list[dict[str, float]]):
    time_ms = [row["time_ms"] for row in rows]
    smallest = values_for_field(rows, "normalized_response_smallest_delay")
    median = values_for_field(rows, "normalized_response_median_delay")
    biggest = values_for_field(rows, "normalized_response_biggest_delay")

    smallest_delay_ms = (
        rows[0].get("delay_ms_marker_smallest", float("nan")) if rows else float("nan")
    )
    median_delay_ms = (
        rows[0].get("delay_ms_marker_median", float("nan")) if rows else float("nan")
    )
    biggest_delay_ms = (
        rows[0].get("delay_ms_marker_biggest", float("nan")) if rows else float("nan")
    )

    figure, ax = plt.subplots(figsize=(11, 6))
    smallest_line = ax.plot(
        time_ms,
        smallest,
        linewidth=1.8,
        color=SMALLEST_COLOR,
        label=(
            f"smallest delay ({smallest_delay_ms:.3f} ms)"
            if not math.isnan(smallest_delay_ms)
            else "smallest delay"
        ),
    )[0]
    median_line = ax.plot(
        time_ms,
        median,
        linewidth=1.8,
        color=MEDIAN_COLOR,
        label=(
            f"median delay ({median_delay_ms:.3f} ms)"
            if not math.isnan(median_delay_ms)
            else "median delay"
        ),
    )[0]
    biggest_line = ax.plot(
        time_ms,
        biggest,
        linewidth=1.8,
        color=BIGGEST_COLOR,
        label=(
            f"biggest delay ({biggest_delay_ms:.3f} ms)"
            if not math.isnan(biggest_delay_ms)
            else "biggest delay"
        ),
    )[0]

    smallest_marker = None
    if not math.isnan(smallest_delay_ms) and smallest_delay_ms >= 0.0:
        smallest_marker = ax.axvline(
            smallest_delay_ms,
            color=SMALLEST_COLOR,
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )
    median_marker = None
    if not math.isnan(median_delay_ms) and median_delay_ms >= 0.0:
        median_marker = ax.axvline(
            median_delay_ms,
            color=MEDIAN_COLOR,
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )
    biggest_marker = None
    if not math.isnan(biggest_delay_ms) and biggest_delay_ms >= 0.0:
        biggest_marker = ax.axvline(
            biggest_delay_ms,
            color=BIGGEST_COLOR,
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized amplitude")
    ax.set_title(csv_path.stem)
    ax.grid(True, alpha=0.3)
    attach_toggle_legend(
        ax,
        [
            {
                "label": (
                    f"smallest delay ({smallest_delay_ms:.3f} ms)"
                    if not math.isnan(smallest_delay_ms)
                    else "smallest delay"
                ),
                "color": SMALLEST_COLOR,
                "lines": [smallest_line, smallest_marker] if smallest_marker else [smallest_line],
            },
            {
                "label": (
                    f"median delay ({median_delay_ms:.3f} ms)"
                    if not math.isnan(median_delay_ms)
                    else "median delay"
                ),
                "color": MEDIAN_COLOR,
                "lines": [median_line, median_marker] if median_marker else [median_line],
            },
            {
                "label": (
                    f"biggest delay ({biggest_delay_ms:.3f} ms)"
                    if not math.isnan(biggest_delay_ms)
                    else "biggest delay"
                ),
                "color": BIGGEST_COLOR,
                "lines": [biggest_line, biggest_marker] if biggest_marker else [biggest_line],
            },
        ],
    )
    figure.tight_layout()
    return figure


def plot_consolidated(
    csv_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, float]],
):
    time_ms = [row["time_ms"] for row in rows]
    response_fields = sorted_response_fields(fieldnames)
    response_series = [values_for_field(rows, field) for field in response_fields]

    smallest_index = None
    biggest_index = None
    selected_delay_profiles = load_selected_delay_profiles(csv_path)
    smallest_profile = selected_delay_profiles.get("smallest")
    biggest_profile = selected_delay_profiles.get("biggest")

    for index, response in enumerate(response_series):
        if smallest_index is None and smallest_profile and series_match(response, smallest_profile):
            smallest_index = index
        if biggest_index is None and biggest_profile and series_match(response, biggest_profile):
            biggest_index = index

    if smallest_index is None or biggest_index is None:
        indexed_delays = []
        for index, response in enumerate(response_series):
            delay_index = infer_delay_index(response)
            if delay_index is not None:
                indexed_delays.append((delay_index, index))
        indexed_delays.sort()

        if indexed_delays:
            if smallest_index is None:
                smallest_index = indexed_delays[0][1]
            if biggest_index is None:
                biggest_index = indexed_delays[-1][1]

    figure, ax = plt.subplots(figsize=(11, 6))
    smallest_lines = []
    biggest_lines = []
    rest_lines = []
    if response_fields:
        response_colors = build_response_colors(len(response_fields))
        for index, response in enumerate(response_series):
            color = response_colors[index]
            linewidth = 1.2
            zorder = 1
            target_group = rest_lines

            if index == smallest_index:
                color = SMALLEST_COLOR
                linewidth = 1.8
                zorder = 3
                target_group = smallest_lines
            elif index == biggest_index:
                color = BIGGEST_COLOR
                linewidth = 1.8
                zorder = 3
                target_group = biggest_lines

            line = ax.plot(
                time_ms,
                response,
                linewidth=linewidth,
                alpha=1.0,
                color=color,
                zorder=zorder,
            )[0]
            target_group.append(line)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized amplitude")
    ax.set_title(csv_path.stem)
    ax.grid(True, alpha=0.3)
    attach_toggle_legend(
        ax,
        [
            {"label": "biggest", "color": BIGGEST_COLOR, "lines": biggest_lines},
            {"label": "smallest", "color": SMALLEST_COLOR, "lines": smallest_lines},
            {"label": "rest", "color": REST_COLOR, "lines": rest_lines},
        ],
    )
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
    if {
        "time_ms",
        "normalized_response_smallest_delay",
        "delay_ms_marker_smallest",
        "normalized_response_median_delay",
        "delay_ms_marker_median",
        "normalized_response_biggest_delay",
        "delay_ms_marker_biggest",
    }.issubset(field_set):
        figure = plot_selected_delays(csv_path, rows)
    elif {"time_ms", "normalized_reference", "normalized_response"}.issubset(field_set):
        figure = plot_single_response(csv_path, rows)
    elif "time_ms" in field_set and "normalized_reference" in field_set:
        figure = plot_consolidated(csv_path, fieldnames, rows)
    else:
        print(f"Unsupported CSV shape for {csv_path}")
        return 1

    output_path = finalize_figure(figure, csv_path)
    print(f"Saved plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

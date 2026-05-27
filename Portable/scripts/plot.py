import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("This plotter needs matplotlib. Install it with: python3 -m pip install matplotlib")
    sys.exit(1)


def load_numeric_csv(csv_path: str | Path) -> tuple[list[str], list[list[float]]]:
    resolved_path = Path(csv_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {resolved_path}")

    with resolved_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError("CSV must contain one x-axis column and at least one y column.")

        columns = [[] for _ in header]
        for row in reader:
            if not row or not row[0].strip():
                continue

            padded = row + [""] * (len(header) - len(row))
            for column_index, value in enumerate(padded[: len(header)]):
                columns[column_index].append(float(value) if value.strip() else float("nan"))

    return header, columns


def plot_numeric_csv(
    csv_path: str | Path,
    *,
    ax=None,
    show: bool = True):
    resolved_path = Path(csv_path).expanduser()
    header, columns = load_numeric_csv(resolved_path)

    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    x_axis = columns[0]

    series_to_plot = list(zip(header[1:], columns[1:]))
    indexed_input_series = []
    for offset, (series_name, series_values) in enumerate(series_to_plot, start=1):
        if not series_name.startswith("input_ch"):
            continue

        sample_count = 0
        mean_square_sum = 0.0
        for value in series_values:
            if value != value:
                continue
            mean_square_sum += value * value
            sample_count += 1

        mean_square = mean_square_sum / sample_count if sample_count else 0.0
        indexed_input_series.append((mean_square, offset, series_name, series_values))

    if indexed_input_series:
        indexed_input_series.sort(key=lambda item: (-item[0], item[1]))
        series_to_plot = [
            (series_name, series_values)
            for _, _, series_name, series_values in indexed_input_series[:5]
        ]
        print(
            "Plotting top input channels by mean square:",
            ", ".join(series_name for series_name, _ in series_to_plot),
        )

    lines = []
    for series_name, series_values in series_to_plot:
        line, = ax.plot(x_axis, series_values, label=series_name)
        lines.append(line)

    ax.set_xlabel(header[0])
    ax.set_ylabel("input amplitude")
    ax.set_title(resolved_path.name)
    ax.grid(True)

    legend = ax.legend()
    linked = {}

    for legend_line, original_line in zip(legend.get_lines(), lines):
        legend_line.set_picker(True)
        linked[legend_line] = original_line

    def on_pick(event):
        legend_line = event.artist
        if legend_line not in linked:
            return

        original_line = linked[legend_line]
        visible = not original_line.get_visible()
        original_line.set_visible(visible)
        legend_line.set_alpha(1.0 if visible else 0.2)
        figure.canvas.draw()

    figure.canvas.mpl_connect("pick_event", on_pick)

    if show:
        plt.show()

    return figure, ax


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python Portable/scripts/plot.py <filename.csv>")
        return 1

    try:
        plot_numeric_csv(sys.argv[1], show=True)
    except (FileNotFoundError, ValueError) as ex:
        print(ex)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

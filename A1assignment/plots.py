import csv
from math import comb
from pathlib import Path


def theoretical_probability(dimensions, patterns):
    terms = sum(comb(patterns - 1, k) for k in range(min(dimensions, patterns)))
    return terms / (2 ** (patterns - 1))


def plot_capacity(csv_paths, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install plotting support with: python -m pip install matplotlib") from error

    by_dimension = {}
    for csv_path in csv_paths:
        with Path(csv_path).open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                dimensions = int(row["N"])
                patterns = int(row["P"])
                by_dimension.setdefault(dimensions, []).append(
                    (patterns, float(row["Q_ls"]))
                )

    figure, axis = plt.subplots(figsize=(8, 5))
    for dimensions, rows in sorted(by_dimension.items()):
        rows.sort()
        patterns = [row[0] for row in rows]
        axis.plot(
            [p / dimensions for p in patterns],
            [row[1] for row in rows],
            marker="o",
            label=f"empirical, N={dimensions}",
        )
        axis.plot(
            [p / dimensions for p in patterns],
            [theoretical_probability(dimensions, p) for p in patterns],
            linestyle="--",
            label=f"theory, N={dimensions}",
        )

    axis.set_xlabel(r"$\alpha=P/N$")
    axis.set_ylabel(r"$Q_{\mathrm{l.s.}}$")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlim(left=0.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_strength(csv_path, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install plotting support with: python -m pip install matplotlib") from error

    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        values = [int(row["x_mu"]) for row in csv.DictReader(file)]
    if not values:
        raise ValueError("No converged-run strengths to plot")

    largest = max(values)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=[value - 0.5 for value in range(largest + 2)], edgecolor="black")
    axis.set_xticks(range(largest + 1))
    axis.set_xlabel("Embedding strength x^μ (updates for pattern μ)")
    axis.set_ylabel("Frequency among patterns in converged runs")
    axis.set_title("Embedding-strength distribution in successful runs")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

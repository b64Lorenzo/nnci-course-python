import csv
from math import comb, erfc, sqrt
from pathlib import Path


def theoretical_probability(dimensions, patterns):
    if dimensions <= 0 or patterns <= 0:
        raise ValueError("N and P must be positive")
    terms = sum(comb(patterns - 1, k) for k in range(min(dimensions, patterns)))
    return terms / (2 ** (patterns - 1))


def normal_approximation(dimensions, patterns):
    if dimensions <= 0 or patterns <= 0:
        raise ValueError("N and P must be positive")
    if patterns == 1:
        return 1.0
    z = (2 * dimensions - patterns) / sqrt(patterns - 1)
    return 0.5 * erfc(-z / sqrt(2.0))


def wilson_interval(successes, datasets):
    if datasets <= 0 or not 0 <= successes <= datasets:
        raise ValueError("successes must be between zero and a positive dataset count")
    z = 1.96
    rate = successes / datasets
    denominator = 1.0 + z * z / datasets
    center = (rate + z * z / (2.0 * datasets)) / denominator
    margin = z * sqrt(
        rate * (1.0 - rate) / datasets + z * z / (4.0 * datasets**2)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def plot_capacity(csv_paths, output_path, clamped=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install plotting support with: python -m pip install matplotlib") from error

    by_dimension = {}
    alpha_max = 2.0
    for csv_path in csv_paths:
        with Path(csv_path).open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                dimensions = int(row["N"])
                patterns = int(row["P"])
                alpha_max = max(alpha_max, patterns / dimensions)
                datasets = int(row["nD"])
                successes = int(row["successes"])
                low, high = wilson_interval(successes, datasets)
                exact = row.get("P_ls_exact")
                normal = row.get("P_ls_normal")
                if not exact:
                    exact = theoretical_probability(dimensions + int(clamped), patterns)
                if not normal:
                    normal = normal_approximation(dimensions + int(clamped), patterns)
                by_dimension.setdefault(dimensions, []).append(
                    (patterns, float(row["Q_ls"]), low, high, float(exact), float(normal))
                )

    if not by_dimension:
        raise ValueError("No capacity results to plot")

    figure, axis = plt.subplots(figsize=(8, 5))
    for dimensions, rows in sorted(by_dimension.items()):
        rows.sort(key=lambda row: row[0])
        patterns = [row[0] for row in rows]
        alpha = [p / dimensions for p in patterns]
        empirical = [row[1] for row in rows]
        lower = [row[2] for row in rows]
        upper = [row[3] for row in rows]
        lower_error = [q - low for q, low in zip(empirical, lower)]
        upper_error = [high - q for q, high in zip(empirical, upper)]
        empirical_line = axis.errorbar(
            alpha,
            empirical,
            yerr=[lower_error, upper_error],
            fmt="o-",
            capsize=3,
            label=f"empirical (95% Wilson), N={dimensions}",
        )
        color = empirical_line.lines[0].get_color()
        axis.plot(
            alpha,
            [row[4] for row in rows],
            color=color,
            linestyle="--",
            label=f"Cover exact, N={dimensions}",
        )
        axis.plot(
            alpha,
            [row[5] for row in rows],
            color=color,
            linestyle=":",
            label=f"CLT approximation, N={dimensions}",
        )

    axis.plot([0.0, 2.0, 2.0, alpha_max], [1.0, 1.0, 0.0, 0.0],
              color="black", linestyle="-.", label="thermodynamic limit ($\\alpha_c=2$)")
    if not clamped:
        axis.scatter([2.0], [0.5], color="black", marker="D", zorder=5,
                     label=r"$P=2N$: $P_{\mathrm{ls}}=1/2$")
    if clamped:
        axis.set_title("Inhomogeneous perceptron (clamped input +1)")
    axis.set_xlabel(r"$\alpha=P/N$")
    axis.set_ylabel(r"Probability ($P_{\mathrm{ls}}$, $Q_{\mathrm{ls}}$)")
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

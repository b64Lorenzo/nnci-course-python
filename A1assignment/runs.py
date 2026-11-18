import csv
from pathlib import Path
from random import Random

from A1assignment.perceptron_rosenblatt import make_random_binary_data, train_rosenblatt
from A1assignment.plots import (
    normal_approximation,
    plot_capacity,
    plot_strength,
    theoretical_probability,
    wilson_interval,
)
from logger import log


def pattern_range(arguments, dimensions):
    step = arguments.p_step if arguments.p_step is not None else max(1, dimensions // 10)
    first = arguments.p_min if arguments.p_min is not None else step
    last = arguments.p_max if arguments.p_max is not None else 3 * dimensions
    if last < first:
        raise ValueError("Pattern range must be ordered")
    return first, last, step


def capacity_rows(dimensions, first, last, step, datasets, max_epochs, seed, threshold):
    random = Random(seed)
    rows = []
    for patterns in range(first, last + 1, step):
        successes = 0
        for _ in range(datasets):
            examples = make_random_binary_data(patterns, dimensions, random.getrandbits(64))
            if train_rosenblatt(examples, max_epochs, threshold)[3]:
                successes += 1
        rate = successes / datasets
        interval = wilson_interval(successes, datasets)
        rows.append({
            "N": dimensions,
            "P": patterns,
            "alpha": patterns / dimensions,
            "nD": datasets,
            "nmax": max_epochs,
            "successes": successes,
            "Q_ls": rate,
            "Q_ls_variance_est": rate * (1.0 - rate) / datasets,
            "P_ls_exact": theoretical_probability(dimensions, patterns),
            "P_ls_normal": normal_approximation(dimensions, patterns),
            "Q_ls_wilson95_low": interval[0],
            "Q_ls_wilson95_high": interval[1],
            "seed": seed,
        })
    return rows


def write_capacity_csv(rows, path):
    fields = (
        "N", "P", "alpha", "nD", "nmax", "successes", "Q_ls",
        "Q_ls_variance_est", "P_ls_exact", "P_ls_normal",
        "Q_ls_wilson95_low", "Q_ls_wilson95_high", "seed",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_capacity(arguments):
    first, last, step = pattern_range(arguments, arguments.dimensions)
    rows = capacity_rows(
        arguments.dimensions, first, last, step, arguments.datasets,
        arguments.max_epochs, arguments.seed, arguments.threshold,
    )
    output = Path(arguments.output or f"results/capacity_N{arguments.dimensions}.csv")
    write_capacity_csv(rows, output)
    plot = output.with_suffix(".png")
    plot_capacity([output], plot)
    print(f"Wrote {len(rows)} capacity points to {output}")
    print(f"Saved capacity plot to {plot}")
    log("INFO", "capacity", f"Wrote {len(rows)} points to {output}")


def run_n_capacity(arguments):
    start, end, step = arguments.NCapacity
    if end < start:
        raise ValueError("NCapacity END must be at least START")

    all_rows = []
    for dimensions in range(start, end + 1, step):
        first, last, p_step = pattern_range(arguments, dimensions)
        rows = capacity_rows(
            dimensions, first, last, p_step, arguments.datasets,
            arguments.max_epochs, arguments.seed, arguments.threshold,
        )
        write_capacity_csv(rows, f"results/capacity_N{dimensions}.csv")
        all_rows.extend(rows)

    output = Path("results/capacity_N_sweep.csv")
    plot = output.with_suffix(".png")
    write_capacity_csv(all_rows, output)
    plot_capacity([output], plot)
    print(f"Wrote {len(all_rows)} capacity points to {output}")
    print(f"Saved combined capacity plot to {plot}")
    log("INFO", "capacity", f"Wrote {len(all_rows)} N-sweep points to {output}")


def run_strength(arguments):
    patterns = arguments.patterns or arguments.dimensions
    random = Random(arguments.seed)
    strengths = []
    for _ in range(arguments.datasets):
        examples = make_random_binary_data(patterns, arguments.dimensions, random.getrandbits(64))
        result = train_rosenblatt(examples, arguments.max_epochs, arguments.threshold)
        if result[3]:
            strengths.append(result[4])
    if not strengths:
        raise ValueError("No datasets converged; there are no strengths to plot")

    output = Path("results/strengths.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("successful_case", "mu", "x_mu"))
        for case, values in enumerate(strengths, start=1):
            writer.writerows((case, pattern, value) for pattern, value in enumerate(values, start=1))

    plot = Path("results/strength_histogram.png")
    plot_strength(output, plot)
    print(f"Converged datasets: {len(strengths)}")
    print(f"Saved strength histogram to {plot}")
    log("INFO", "strength", f"Converged datasets: {len(strengths)}")


def run_single(arguments):
    patterns = arguments.patterns or arguments.dimensions
    result = train_rosenblatt(
        make_random_binary_data(patterns, arguments.dimensions, arguments.seed),
        arguments.max_epochs,
        arguments.threshold,
    )
    status = "Converged" if result[3] else "Did not converge"
    message = f"{status} after {result[1]} epochs; updates={result[2]}"
    print(message)
    log("INFO", "perceptron", message)


def run_perceptron(arguments):
    if arguments.NCapacity:
        run_n_capacity(arguments)
    elif arguments.capacity:
        run_capacity(arguments)
    elif arguments.strength:
        run_strength(arguments)
    else:
        run_single(arguments)

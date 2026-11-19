import csv
from math import comb
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


def resolve_max_epochs(dimensions, datasets, max_epochs):
    return dimensions * datasets if max_epochs is None else max_epochs


def fisher_exact_two_sided(table):
    (a, b), (c, d) = table
    if any(type(value) is not int or value < 0 for value in (a, b, c, d)):
        raise ValueError("Fisher counts must be non-negative integers")
    first, second, successes = a + b, c + d, a + c
    observed = comb(first, a) * comb(second, c)
    weights = (comb(first, k) * comb(second, successes - k)
               for k in range(max(0, successes - second), min(first, successes) + 1))
    # The common denominator lets us compare probabilities using exact integers.
    return sum(weight for weight in weights if weight <= observed) / comb(first + second, successes)


def capacity_rows(dimensions, first, last, step, datasets, max_epochs, seed, threshold, clamped=False):
    max_epochs = resolve_max_epochs(dimensions, datasets, max_epochs)
    model_dimensions = dimensions + int(clamped)
    random = Random(seed)
    rows = []
    for patterns in range(first, last + 1, step):
        successes = 0
        for _ in range(datasets):
            examples = make_random_binary_data(patterns, dimensions, random.getrandbits(64))
            if train_rosenblatt(examples, max_epochs, threshold, clamped=clamped)[3]:
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
            "P_ls_exact": theoretical_probability(model_dimensions, patterns),
            "P_ls_normal": normal_approximation(model_dimensions, patterns),
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
        arguments.max_epochs, arguments.seed, arguments.threshold, arguments.clamped,
    )
    output = Path(arguments.output or f"results/capacity_N{arguments.dimensions}.csv")
    write_capacity_csv(rows, output)
    plot = output.with_suffix(".png")
    plot_capacity([output], plot, clamped=arguments.clamped)
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
            arguments.max_epochs, arguments.seed, arguments.threshold, arguments.clamped,
        )
        write_capacity_csv(rows, f"results/capacity_N{dimensions}.csv")
        all_rows.extend(rows)

    output = Path("results/capacity_N_sweep.csv")
    plot = output.with_suffix(".png")
    write_capacity_csv(all_rows, output)
    plot_capacity([output], plot, clamped=arguments.clamped)
    print(f"Wrote {len(all_rows)} capacity points to {output}")
    print(f"Saved combined capacity plot to {plot}")
    log("INFO", "capacity", f"Wrote {len(all_rows)} N-sweep points to {output}")


def run_strength(arguments):
    max_epochs = resolve_max_epochs(arguments.dimensions, arguments.datasets, arguments.max_epochs)
    patterns = arguments.patterns or arguments.dimensions
    random = Random(arguments.seed)
    strengths = []
    for _ in range(arguments.datasets):
        examples = make_random_binary_data(patterns, arguments.dimensions, random.getrandbits(64))
        result = train_rosenblatt(examples, max_epochs, arguments.threshold, clamped=arguments.clamped)
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
        resolve_max_epochs(arguments.dimensions, arguments.datasets, arguments.max_epochs),
        arguments.threshold,
        clamped=arguments.clamped,
    )
    status = "Converged" if result[3] else "Did not converge"
    message = f"{status} after {result[1]} epochs; updates={result[2]}"
    print(message)
    log("INFO", "perceptron", message)


def run_clamped_test(arguments):
    patterns = arguments.patterns or 2 * arguments.dimensions
    rows = [capacity_rows(
        arguments.dimensions, patterns, patterns, 1, arguments.datasets,
        arguments.max_epochs, arguments.seed + int(clamped), arguments.threshold, clamped,
    )[0] for clamped in (False, True)]
    p_value = fisher_exact_two_sided([
        (row["successes"], row["nD"] - row["successes"]) for row in rows
    ])
    summary = {key: rows[0][key] for key in ("N", "P", "nD", "nmax")}
    summary["threshold"] = arguments.threshold
    for name, row in zip(("homogeneous", "clamped"), rows):
        summary.update({f"{name}_{key}": row[key] for key in ("seed", "successes")})
        summary[f"{name}_rate"] = row["Q_ls"]
        print(f"{name.capitalize()}: {row['successes']}/{row['nD']} "
              f"({row['Q_ls']:.2%}), seed={row['seed']}")
    summary["rate_difference"] = rows[1]["Q_ls"] - rows[0]["Q_ls"]
    summary["fisher_p_two_sided"] = p_value
    summary["significant_0_05"] = p_value < 0.05
    print(f"Difference: {100 * summary['rate_difference']:+.2f} percentage points")
    print(f"Two-sided Fisher exact p-value: {p_value:.12g}")
    print(f"Significant at 0.05: {'yes' if p_value < 0.05 else 'no'}")
    output = Path(arguments.output or "results/clamped_test.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(f"Wrote clamped comparison to {output}")


def run_perceptron(arguments):
    if arguments.clampedTest:
        run_clamped_test(arguments)
    elif arguments.NCapacity:
        run_n_capacity(arguments)
    elif arguments.capacity:
        run_capacity(arguments)
    elif arguments.strength:
        run_strength(arguments)
    else:
        run_single(arguments)

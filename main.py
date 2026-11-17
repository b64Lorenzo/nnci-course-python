import argparse
import csv
import math
from pathlib import Path
from random import Random

from A1assignment.perceptron_rosenblatt import make_random_binary_data, train_rosenblatt
from A1assignment.plots import plot_capacity, plot_strength


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def make_parser():
    parser = argparse.ArgumentParser(description="Run the Assignment I perceptron project.")
    parser.add_argument("command", nargs="?", choices=("perceptron",))
    parser.add_argument("--dimensions", type=positive_int, default=20, help="feature count (default: 20)")
    parser.add_argument("--patterns", type=positive_int, help="patterns for a single run or strength study (default: N)")
    parser.add_argument("--datasets", type=positive_int, default=50, help="datasets per P or strength trials (default: 50)")
    parser.add_argument("--max-epochs", type=positive_int, default=100, help="maximum training sweeps (default: 100)")
    parser.add_argument("--seed", type=int, default=2026, help="random seed (default: 2026)")
    parser.add_argument("--threshold", type=float, default=0.0, help="update threshold (default: 0)")
    study = parser.add_mutually_exclusive_group()
    study.add_argument("--capacity", action="store_true", help="run a capacity study over P")
    study.add_argument("--NCapacity", type=positive_int, nargs=3, metavar=("START", "END", "STEP"), help="sweep feature count N")
    parser.add_argument("--strength", action="store_true", help="plot strengths from converged trials")
    parser.add_argument("--p-min", type=positive_int, help="first pattern count in a capacity study")
    parser.add_argument("--p-max", type=positive_int, help="last pattern count in a capacity study")
    parser.add_argument("--p-step", type=positive_int, help="pattern-count step in a capacity study")
    parser.add_argument("--output", help="capacity CSV path (default: results/capacity_N<N>.csv)")
    return parser


def pattern_range(arguments, dimensions):
    step = arguments.p_step or max(1, dimensions // 10)
    first = arguments.p_min or step
    last = arguments.p_max or 3 * dimensions
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
        rows.append({
            "N": dimensions,
            "P": patterns,
            "alpha": patterns / dimensions,
            "nD": datasets,
            "nmax": max_epochs,
            "successes": successes,
            "Q_ls": successes / datasets,
            "seed": seed,
        })
    return rows


def write_capacity_csv(rows, path):
    fields = ("N", "P", "alpha", "nD", "nmax", "successes", "Q_ls", "seed")
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


def main():
    parser = make_parser()
    arguments = parser.parse_args()
    if arguments.command is None:
        parser.print_help()
        return 0
    if not math.isfinite(arguments.threshold):
        parser.error("--threshold must be finite")
    if arguments.strength and (arguments.capacity or arguments.NCapacity):
        parser.error("Use --strength separately from capacity options")

    try:
        if arguments.NCapacity:
            run_n_capacity(arguments)
        elif arguments.capacity:
            run_capacity(arguments)
        elif arguments.strength:
            run_strength(arguments)
        else:
            patterns = arguments.patterns or arguments.dimensions
            result = train_rosenblatt(
                make_random_binary_data(patterns, arguments.dimensions, arguments.seed),
                arguments.max_epochs,
                arguments.threshold,
            )
            status = "Converged" if result[3] else "Did not converge"
            print(f"{status} after {result[1]} epochs; updates={result[2]}")
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

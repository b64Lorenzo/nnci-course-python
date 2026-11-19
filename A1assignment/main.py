import argparse
import math

from A1assignment.runs import run_perceptron
from logger import log


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def make_parser():
    parser = argparse.ArgumentParser(description="Run Assignment I: the Rosenblatt perceptron.")
    parser.add_argument("command", nargs="?", choices=("perceptron",))
    parser.add_argument("--dimensions", type=positive_int, default=20, help="feature count (default: 20)")
    parser.add_argument("--clamped", action="store_true", help="append a +1 bias input (inhomogeneous perceptron)")
    parser.add_argument("--patterns", type=positive_int, help="pattern count (default: N, or 2N for --clampedTest)")
    parser.add_argument("--datasets", type=positive_int, default=50, help="datasets per P or strength trials (default: 50)")
    parser.add_argument("--max-epochs", type=positive_int, default=None, help="maximum training sweeps (default: N * nD, dimensions * datasets)")
    parser.add_argument("--seed", type=int, default=2026, help="random seed (default: 2026)")
    parser.add_argument("--threshold", type=float, default=0.0, help="update threshold (default: 0)")
    study = parser.add_mutually_exclusive_group()
    study.add_argument("--capacity", action="store_true", help="run a capacity study over P")
    study.add_argument("--clampedTest", action="store_true", help="compare homogeneous/clamped success rates with a two-sided Fisher exact test")
    study.add_argument("--NCapacity", type=positive_int, nargs=3, metavar=("START", "END", "STEP"), help="sweep feature count N")
    parser.add_argument("--strength", action="store_true", help="plot strengths from converged trials")
    parser.add_argument("--p-min", type=positive_int, help="first pattern count in a capacity study")
    parser.add_argument("--p-max", type=positive_int, help="last pattern count in a capacity study")
    parser.add_argument("--p-step", type=positive_int, help="pattern-count step in a capacity study")
    parser.add_argument("--output", help="study CSV path (default: per-study path in results/)")
    return parser


def main(argv=None):
    parser = make_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 0
    if not math.isfinite(arguments.threshold):
        parser.error("--threshold must be finite")
    if arguments.strength and (arguments.capacity or arguments.NCapacity or arguments.clampedTest):
        parser.error("Use --strength separately from other study options")
    if arguments.clampedTest and arguments.clamped:
        parser.error("--clampedTest already compares both cases; omit --clamped")
    if arguments.clampedTest and any(value is not None for value in
                                     (arguments.p_min, arguments.p_max, arguments.p_step)):
        parser.error("--clampedTest uses --patterns, not a P sweep")

    try:
        log("INFO", "app", f"command started: {arguments.command}")
        run_perceptron(arguments)
        log("INFO", "app", f"command completed: {arguments.command}")
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

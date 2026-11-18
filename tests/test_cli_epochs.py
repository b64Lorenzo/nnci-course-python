import csv
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from A1assignment.main import make_parser
from A1assignment.runs import resolve_max_epochs


CLI = Path(__file__).resolve().parents[1] / "main.py"


class EpochOptionTests(unittest.TestCase):
    def test_omitted_maximum_is_resolved_from_dimensions_and_datasets(self):
        arguments = make_parser().parse_args(["perceptron"])
        self.assertIsNone(arguments.max_epochs)
        self.assertEqual(resolve_max_epochs(arguments.dimensions, arguments.datasets, None), 1000)
        self.assertIn("default: N * nD, dimensions * datasets", " ".join(make_parser().format_help().split()))

    def test_explicit_maximum_is_not_rescaled(self):
        for maximum in (1, 7, 100):
            with self.subTest(maximum=maximum):
                self.assertEqual(resolve_max_epochs(2, 3, maximum), maximum)


class CliEpochTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="nnci-cli-")
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def run_cli(self, *arguments, expected_code=0):
        result = subprocess.run(
            [sys.executable, str(CLI), "perceptron", *arguments],
            cwd=self.directory,
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
                 "MPLCONFIGDIR": str(self.directory / ".mplconfig")},
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
        return result

    def read_csv(self, path):
        with (self.directory / path).open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_single_default_exhausts_dimensions_times_datasets_epochs(self):
        result = self.run_cli("--dimensions", "2", "--datasets", "3", "--patterns", "20")
        self.assertIn("Did not converge after 6 epochs;", result.stdout)

    def test_single_explicit_maximum_is_preserved(self):
        result = self.run_cli(
            "--dimensions", "2", "--datasets", "3", "--patterns", "20", "--max-epochs", "1",
        )
        self.assertIn("Did not converge after 1 epochs;", result.stdout)

    def test_strength_default_filters_trials_at_dimensions_times_datasets_epochs(self):
        result = self.run_cli(
            "--strength", "--dimensions", "2", "--datasets", "3",
            "--patterns", "4", "--seed", "32",
        )
        self.assertIn("Converged datasets: 2", result.stdout)
        rows = self.read_csv("results/strengths.csv")
        self.assertEqual(
            [(int(row["successful_case"]), int(row["mu"]), int(row["x_mu"])) for row in rows],
            [(case, mu, value) for case, values in enumerate(((1, 1, 1, 0), (6, 0, 3, 2)), 1)
             for mu, value in enumerate(values, 1)],
        )
        self.assertTrue((self.directory / "results/strength_histogram.png").is_file())

    def test_strength_explicit_maximum_filters_trials_at_that_limit(self):
        result = self.run_cli(
            "--strength", "--dimensions", "2", "--datasets", "3",
            "--patterns", "4", "--seed", "32", "--max-epochs", "1",
        )
        self.assertIn("Converged datasets: 1", result.stdout)
        rows = self.read_csv("results/strengths.csv")
        self.assertEqual([int(row["x_mu"]) for row in rows], [1, 1, 1, 0])

    def test_capacity_default_uses_dimensions_times_datasets_not_patterns(self):
        self.run_cli(
            "--capacity", "--dimensions", "2", "--datasets", "3",
            "--p-min", "4", "--p-max", "5", "--p-step", "1", "--seed", "32",
        )
        rows = self.read_csv("results/capacity_N2.csv")
        self.assertEqual([(int(row["P"]), int(row["nmax"])) for row in rows], [(4, 6), (5, 6)])
        self.assertEqual(int(rows[0]["successes"]), 2)
        self.assertTrue((self.directory / "results/capacity_N2.png").is_file())

    def test_capacity_explicit_maximum_matches_training_and_csv(self):
        self.run_cli(
            "--capacity", "--dimensions", "2", "--datasets", "3", "--max-epochs", "1",
            "--p-min", "4", "--p-max", "5", "--p-step", "1", "--seed", "32",
        )
        rows = self.read_csv("results/capacity_N2.csv")
        self.assertEqual([(int(row["P"]), int(row["nmax"])) for row in rows], [(4, 1), (5, 1)])
        self.assertEqual(int(rows[0]["successes"]), 1)

    def test_n_sweep_recomputes_default_maximum_for_each_dimension(self):
        self.run_cli("--NCapacity", "2", "4", "2", "--dimensions", "17", "--datasets", "3")
        rows = self.read_csv("results/capacity_N_sweep.csv")
        self.assertEqual(
            [(int(row["N"]), int(row["P"]), int(row["nmax"])) for row in rows],
            [(n, p, n * 3) for n in (2, 4) for p in range(1, 3 * n + 1)],
        )
        for n in (2, 4):
            self.assertEqual(self.read_csv(f"results/capacity_N{n}.csv"),
                             [row for row in rows if int(row["N"]) == n])
        self.assertTrue((self.directory / "results/capacity_N_sweep.png").is_file())

    def test_n_sweep_preserves_explicit_maximum_and_pattern_bounds(self):
        self.run_cli(
            "--NCapacity", "2", "4", "2", "--datasets", "3", "--max-epochs", "7",
            "--p-min", "3", "--p-max", "4", "--p-step", "1",
        )
        rows = self.read_csv("results/capacity_N_sweep.csv")
        self.assertEqual(
            [(int(row["N"]), int(row["P"]), int(row["nmax"])) for row in rows],
            [(2, 3, 7), (2, 4, 7), (4, 3, 7), (4, 4, 7)],
        )

    def test_nonpositive_epoch_factors_and_overrides_are_rejected(self):
        for option in ("--dimensions", "--datasets", "--max-epochs", "--NCapacity"):
            for value in ("0", "-1"):
                with self.subTest(option=option, value=value):
                    extra = ("4", "1") if option == "--NCapacity" else ()
                    result = self.run_cli(option, value, *extra, expected_code=2)
                    self.assertIn(f"argument {option}: value must be positive", result.stderr)
                    self.assertFalse((self.directory / "results").exists())
                    self.assertFalse((self.directory / "logs").exists())


if __name__ == "__main__":
    unittest.main()

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

    def test_clamped_capacity_counts_affine_solutions(self):
        arguments = ("--capacity", "--dimensions", "1", "--datasets", "3", "--max-epochs", "1000",
                     "--p-min", "2", "--p-max", "2", "--seed", "17")
        self.run_cli(*arguments)
        self.assertEqual(int(self.read_csv("results/capacity_N1.csv")[0]["successes"]), 1)

        self.run_cli(*arguments, "--clamped")
        row = self.read_csv("results/capacity_N1.csv")[0]
        self.assertEqual(int(row["successes"]), 3)
        self.assertEqual(float(row["Q_ls"]), 1.0)
        self.assertEqual(float(row["P_ls_exact"]), 1.0)
        self.assertEqual((int(row["N"]), int(row["nmax"])), (1, 1000))
        self.assertTrue((self.directory / "results/capacity_N1.png").is_file())

    def test_clamped_single_run_learns_a_bias(self):
        result = self.run_cli("--clamped", "--dimensions", "1", "--patterns", "2",
                              "--seed", "1", "--max-epochs", "1000")
        self.assertEqual(result.stdout.strip(), "Converged after 27 epochs; updates=52")

    def test_clamped_strength_counts_all_affine_solutions(self):
        result = self.run_cli("--clamped", "--strength", "--dimensions", "1", "--patterns", "2",
                              "--datasets", "3", "--seed", "17", "--max-epochs", "1000")
        self.assertIn("Converged datasets: 3", result.stdout)
        rows = self.read_csv("results/strengths.csv")
        self.assertEqual({int(row["successful_case"]) for row in rows}, {1, 2, 3})
        self.assertEqual(len(rows), 3 * 2)
        self.assertTrue((self.directory / "results/strength_histogram.png").is_file())

    def test_clamped_n_sweep_uses_affine_training_and_theory(self):
        self.run_cli("--clamped", "--NCapacity", "1", "2", "1", "--p-min", "2", "--p-max", "2",
                     "--datasets", "3", "--seed", "17", "--max-epochs", "1000")
        rows = self.read_csv("results/capacity_N_sweep.csv")
        self.assertEqual([int(row["N"]) for row in rows], [1, 2])
        self.assertEqual(int(rows[0]["successes"]), 3)
        self.assertEqual(float(rows[0]["P_ls_exact"]), 1.0)
        self.assertTrue((self.directory / "results/capacity_N_sweep.png").is_file())

    def test_clamped_test_reports_independent_groups(self):
        arguments = ("--clampedTest", "--dimensions", "1", "--patterns", "2", "--datasets", "3",
                     "--max-epochs", "1000", "--seed", "17", "--output", "results/comparison.csv")
        result = self.run_cli(*arguments)
        self.assertIn("Homogeneous: 1/3", result.stdout)
        self.assertIn("Clamped: 3/3", result.stdout)
        self.assertIn("Two-sided Fisher exact p-value: 0.4", result.stdout)
        self.assertIn("Significant at 0.05: no", result.stdout)
        row = self.read_csv("results/comparison.csv")[0]
        self.assertEqual((int(row["homogeneous_seed"]), int(row["clamped_seed"])), (17, 18))
        self.assertEqual((int(row["homogeneous_successes"]), int(row["clamped_successes"])), (1, 3))
        self.assertEqual(float(row["homogeneous_rate"]), 1 / 3)
        self.assertEqual(float(row["clamped_rate"]), 1.0)
        self.assertAlmostEqual(float(row["fisher_p_two_sided"]), 0.4)
        before = (self.directory / "results/comparison.csv").read_bytes()
        self.run_cli(*arguments)
        self.assertEqual((self.directory / "results/comparison.csv").read_bytes(), before)
        self.assertEqual(list((self.directory / "results").glob("*.png")), [])

    def test_clamped_test_defaults_to_twice_the_feature_count(self):
        self.run_cli("--clampedTest", "--dimensions", "1", "--datasets", "3", "--seed", "17")
        row = self.read_csv("results/clamped_test.csv")[0]
        self.assertEqual((int(row["N"]), int(row["P"]), int(row["nD"]), int(row["nmax"])), (1, 2, 3, 3))

    def test_clamped_test_rejects_conflicting_modes_and_sweeps(self):
        for extra in (("--capacity",), ("--NCapacity", "1", "2", "1"),
                      ("--strength",), ("--clamped",), ("--p-min", "2")):
            with self.subTest(extra=extra):
                result = self.run_cli("--clampedTest", *extra, expected_code=2)
                self.assertNotIn("unrecognized arguments", result.stderr)
                self.assertFalse((self.directory / "results").exists())
                self.assertFalse((self.directory / "logs").exists())

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

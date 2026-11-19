import unittest

from A1assignment.runs import fisher_exact_two_sided


class FisherExactTests(unittest.TestCase):
    def test_documented_example_and_observed_study(self):
        self.assertAlmostEqual(fisher_exact_two_sided(((6, 2), (1, 4))), 4 / 39)
        self.assertAlmostEqual(
            fisher_exact_two_sided(((224, 276), (266, 234))), 0.009463834038967672,
        )

    def test_degenerate_tables_and_symmetry(self):
        for table in (((0, 0), (0, 0)), ((5, 0), (5, 0)), ((0, 5), (0, 5)), ((2, 3), (2, 3))):
            with self.subTest(table=table):
                self.assertEqual(fisher_exact_two_sided(table), 1.0)
        table = ((6, 2), (1, 4))
        self.assertEqual(fisher_exact_two_sided(table), fisher_exact_two_sided(table[::-1]))
        self.assertEqual(fisher_exact_two_sided(table),
                         fisher_exact_two_sided(tuple(row[::-1] for row in table)))

    def test_invalid_counts_are_rejected(self):
        for value in (-1, 0.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fisher_exact_two_sided(((1, value), (2, 3)))


if __name__ == "__main__":
    unittest.main()

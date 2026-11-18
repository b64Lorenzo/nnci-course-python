import unittest

from A1assignment.perceptron_rosenblatt import train_rosenblatt


class PerceptronValidationTests(unittest.TestCase):
    def test_hand_calculated_separable_case(self):
        examples = [([1.0, 0.0], 1), ([0.0, 1.0], -1)]

        weights, epochs, updates, converged, strengths = train_rosenblatt(
            examples, max_epochs=10, threshold=0.0
        )

        self.assertEqual(weights, [0.5, -0.5])
        self.assertEqual(strengths, [1, 1])
        self.assertEqual(updates, 2)
        self.assertEqual(epochs, 1)
        self.assertTrue(converged)

    def test_opposite_labels_on_same_input_are_not_separable(self):
        examples = [([1.0, 0.0], 1), ([1.0, 0.0], -1)]

        _, _, _, converged, _ = train_rosenblatt(
            examples, max_epochs=2, threshold=0.0
        )

        self.assertFalse(converged)


if __name__ == "__main__":
    unittest.main()

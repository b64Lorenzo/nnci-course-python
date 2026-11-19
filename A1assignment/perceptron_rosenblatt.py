from math import isfinite
from random import Random


def make_random_binary_data(count, dimensions, seed):
    if count < 1 or dimensions < 1:
        raise ValueError("Count and dimensions must be positive")

    random = Random(seed)
    return [
        ([random.gauss(0.0, 1.0) for _ in range(dimensions)], random.choice((-1, 1)))
        for _ in range(count)
    ]


def train_rosenblatt(examples, max_epochs=100, threshold=0.0, clamped=False):
    if max_epochs < 1 or not isfinite(threshold):
        raise ValueError("Maximum epochs must be positive and threshold must be finite")
    if not examples or not examples[0][0]:
        raise ValueError("Training data must not be empty")

    dimensions = len(examples[0][0])
    for inputs, label in examples:
        if len(inputs) != dimensions or label not in (-1, 1):
            raise ValueError("Inputs must have equal dimensions and labels must be -1 or +1")
        if any(not isfinite(value) for value in inputs):
            raise ValueError("Inputs must be finite")

    if clamped:
        examples = [([*inputs, 1.0], label) for inputs, label in examples]
        dimensions += 1

    weights = [0.0] * dimensions
    strengths = [0] * len(examples)
    updates = 0

    for epoch in range(max_epochs):
        for index, (inputs, label) in enumerate(examples):
            potential = label * sum(weight * value for weight, value in zip(weights, inputs))
            if potential <= threshold:
                for feature in range(dimensions):
                    weights[feature] += label * inputs[feature] / dimensions
                strengths[index] += 1
                updates += 1

        converged = all(
            label * sum(weight * value for weight, value in zip(weights, inputs)) > 0.0
            for inputs, label in examples
        )
        if converged:
            return weights, epoch + 1, updates, True, strengths

    return weights, max_epochs, updates, False, strengths

import math
import random
from typing import List, Tuple


def _dot(a: List[float], b: List[float]) -> float:
	return sum(x * y for x, y in zip(a, b))


def _sigmoid(x: float) -> float:
	return 1.0 / (1.0 + math.exp(-x))


def single_neuron(inputs: List[float], weights: List[float], bias: float = 0.0, activation: str = "sigmoid") -> float:
	"""
	Compute a single neuron's output.

	Parameters:
	- inputs: list of input feature values
	- weights: list of weights (same length as inputs)
	- bias: bias term (float)
	- activation: 'sigmoid' or 'step'

	Returns:
	- activated output (float). For 'step' returns 0.0 or 1.0.
	"""
	z = _dot(inputs, weights) + bias
	if activation == "sigmoid":
		return _sigmoid(z)
	elif activation == "step":
		return 1.0 if z > 0.0 else 0.0
	else:
		raise ValueError("activation must be 'sigmoid' or 'step'")


def binary_classifier(weights: List[float], dataset: List[List[float]], bias: float = 0.0, activation: str = "sigmoid") -> List[int]:
	"""
	Apply a single-neuron binary classifier to a dataset.

	Parameters:
	- weights: list of weights
	- dataset: list of feature vectors (each a list of floats)
	- bias: bias term
	- activation: 'sigmoid' or 'step'

	Returns:
	- list of predicted labels (0 or 1)
	"""
	preds = []
	for x in dataset:
		out = single_neuron(x, weights, bias=bias, activation=activation)
		if activation == "sigmoid":
			preds.append(1 if out >= 0.5 else 0)
		else:
			preds.append(int(out))
	return preds


def create_synthetic_dataset(n_samples: int = 100, n_features: int = 2, separation: float = 1.0, random_seed: int = None) -> Tuple[List[List[float]], List[int]]:
	"""
	Create a linearly separable synthetic dataset.

	Returns (X, y) where X is a list of feature lists and y is a list of 0/1 labels.
	"""
	if random_seed is not None:
		random.seed(random_seed)

	# choose a random true weight vector and bias to define the separating hyperplane
	true_w = [random.uniform(-1, 1) for _ in range(n_features)]
	true_b = random.uniform(-0.5, 0.5)

	X = []
	y = []
	for _ in range(n_samples):
		x = [random.gauss(0, 1) + separation * (1 if random.random() > 0.5 else -1) for _ in range(n_features)]
		label = 1 if _dot(x, true_w) + true_b > 0 else 0
		X.append(x)
		y.append(label)
	return X, y


def calculate_weights(dataset: List[List[float]], labels: List[int], activation: str = "step", learning_rate: float = 0.1, epochs: int = 100) -> Tuple[List[float], float]:
	"""
	Calculate weights for a binary classifier using either perceptron (for 'step')
	or logistic regression gradient descent (for 'sigmoid').

	Parameters:
	- dataset: list of feature vectors
	- labels: list of 0/1 labels
	- activation: 'step' for perceptron, 'sigmoid' for logistic gradient descent
	- learning_rate: learning rate for updates
	- epochs: number of passes over the data

	Returns:
	- (weights list, bias float)
	"""
	n_features = len(dataset[0])
	weights = [0.0] * n_features
	bias = 0.0

	if activation == "step":
		# Perceptron learning rule
		for _ in range(epochs):
			for x, y in zip(dataset, labels):
				pred = 1 if _dot(x, weights) + bias > 0 else 0
				error = y - pred
				if error != 0:
					for i in range(n_features):
						weights[i] += learning_rate * error * x[i]
					bias += learning_rate * error
		return weights, bias

	elif activation == "sigmoid":
		# Logistic regression via gradient descent
		for _ in range(epochs):
			grad_w = [0.0] * n_features
			grad_b = 0.0
			for x, y in zip(dataset, labels):
				z = _dot(x, weights) + bias
				p = _sigmoid(z)
				error = p - y
				for i in range(n_features):
					grad_w[i] += error * x[i]
				grad_b += error
			# update weights (average gradient)
			m = len(dataset)
			for i in range(n_features):
				weights[i] -= learning_rate * (grad_w[i] / m)
			bias -= learning_rate * (grad_b / m)
		return weights, bias

	else:
		raise ValueError("activation must be 'step' or 'sigmoid'")


def _accuracy(trues: List[int], preds: List[int]) -> float:
	correct = sum(1 for t, p in zip(trues, preds) if t == p)
	return correct / len(trues) if trues else 0.0


if __name__ == "__main__":
	# demo: create data, train with perceptron and logistic, evaluate
	X, y = create_synthetic_dataset(n_samples=200, n_features=2, separation=0.8, random_seed=42)

	# Perceptron (step)
	w_p, b_p = calculate_weights(X, y, activation="step", learning_rate=0.1, epochs=20)
	preds_p = binary_classifier(w_p, X, bias=b_p, activation="step")
	print(f"Perceptron accuracy: {_accuracy(y, preds_p):.3f}")

	# Logistic (sigmoid)
	w_l, b_l = calculate_weights(X, y, activation="sigmoid", learning_rate=0.5, epochs=200)
	preds_l = binary_classifier(w_l, X, bias=b_l, activation="sigmoid")
	print(f"Logistic accuracy:  {_accuracy(y, preds_l):.3f}")


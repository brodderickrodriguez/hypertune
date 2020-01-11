# Brodderick Rodriguez
# 11 Nov 2019

import numpy as np
import hypertune as ht
import datasets


class Perceptron:
	def __init__(self, alpha, epochs=10**3):
		self.alpha = alpha
		self.epochs = epochs

	def fit(self, X, y):
		self.w_ = np.random.rand(1 + X.shape[1])

		for _ in range(self.epochs):
			for xi, yi in zip(X, y):				
				update = self.alpha * (yi - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update

	def predict(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def mse(self, X, y):
		y_hat = self.predict(X)
		mse = np.mean((y - y_hat) ** 2)
		return mse


def aux_objective_func(algo, X, y):
	_y = np.where(y >= 0.0, 1, -1)
	y_hat = np.where(algo.predict(X) >= 0.0, 1, -1)
	return np.mean(_y == y_hat)


X, y = datasets.iris(return_splits=False)
X = X[y != 2]
y = y[y != 2]
y[y == 0] = -1
X_train, X_test, y_train, y_test = datasets.split(X, y)

alpha = ht.ContinuousParameter('alpha', lower_bound=10**-10, upper_bound=0.01)
epochs = ht.DiscreteParameter('epochs', lower_bound=200, upper_bound=10**4)

hypers = [alpha, epochs]

tuner = ht.HyperTune(algorithm=Perceptron,
					parameters=hypers,
					train_func=Perceptron.fit,
					objective_func=Perceptron.mse,
					train_func_args=(X_train, y_train), 
					objective_func_args=(X_test, y_test),
					max_evals=50,
					maximize=False)

results = tuner.tune()
print(results)

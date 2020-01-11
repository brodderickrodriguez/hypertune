# Brodderick Rodriguez
# 18 Nov 2019

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import numpy as np
import hypertune as ht
import datasets


def train(algo):
	pass


def objective_func(algo, X, y, splits):
	results = np.zeros(len(splits))

	for i, (train_idxs, test_idxs) in enumerate(splits):
		X_train, y_train = X[train_idxs], y[train_idxs]
		X_test, y_test = X[test_idxs], y[test_idxs]

		algo.fit(X_train, y_train)
		results[i] = algo.score(X_test, y_test)

	return np.mean(results)


X, y = datasets.iris(return_splits=False)
splits = list(KFold(n_splits=4).split(X))


learning_rate = ht.CategoricalParameter('learning_rate', options=('constant', 'invscaling', 'adaptive'))
learning_rate_init = ht.ContinuousParameter('learning_rate_init', lower_bound=10**-5, upper_bound=0.1)
max_iter = ht.DiscreteParameter('max_iter', lower_bound=500, upper_bound=10**3)

hypers = [learning_rate, learning_rate_init, max_iter]

tuner = ht.HyperTune(algorithm=MLPClassifier,
					parameters=hypers, 
					train_func=train,
					objective_func=objective_func,
					objective_func_args=(X, y, splits),
					max_evals=10**2)

results = tuner.tune()
print(results)

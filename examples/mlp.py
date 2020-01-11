# Brodderick Rodriguez
# 10 Nov 2019

from sklearn.neural_network import MLPRegressor
import numpy as np
import hypertune as ht
import datasets


X_train, X_test, y_train, y_test = datasets.iris()

activation = ht.CategoricalParameter('activation', options=('identity', 'logistic', 'tanh', 'relu'))
solver = ht.CategoricalParameter('solver', options=('lbfgs', 'sgd', 'adam'))
alpha = ht.ContinuousParameter('alpha', lower_bound=10**-10, upper_bound=0.1)
learning_rate = ht.CategoricalParameter('learning_rate', options=('constant', 'invscaling', 'adaptive'))
learning_rate_init = ht.ContinuousParameter('learning_rate_init', lower_bound=10**-5, upper_bound=0.1)
max_iter = ht.DiscreteParameter('max_iter', lower_bound=500, upper_bound=10**3)

hl1 = ht.DiscreteParameter('', lower_bound=50, upper_bound=250)
hl2 = ht.DiscreteParameter('', lower_bound=100, upper_bound=250)
hl3 = ht.DiscreteParameter('', lower_bound=1, upper_bound=100)
hidden_layer_sizes = ht.TupleParameter('hidden_layer_sizes', values=(hl1, hl2, hl3))

hypers = [activation, alpha, learning_rate_init, max_iter, solver, learning_rate, hidden_layer_sizes]


def aux_obj_acc_func(algo, X, y):
	return algo.score(X, y)


def aux_obj_mse_func(algo, X, y):
	y_hat = algo.predict(X)
	return np.mean((y - y_hat) ** 2)


tuner = ht.HyperTune(algorithm=MLPRegressor,
					parameters=hypers, 
					train_func=MLPRegressor.fit,
					objective_func=aux_obj_mse_func,
					train_func_args=(X_train, y_train), 
					objective_func_args=(X_test, y_test),
					max_evals=10**2,
					maximize=False,
					num_replications=1)

results = tuner.tune()
print(results)

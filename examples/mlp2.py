from sklearn.neural_network import MLPRegressor
import hypertune as ht
import numpy as np

# make an example dataset
p = 75
X = np.random.rand(100, 3)
y = np.random.rand(100)
X_train, X_test, y_train, y_test = X[:p], X[p:], y[:p], y[p:]

# define the target hyperparameters
activation = ht.CategoricalParameter('activation', options=('identity', 'logistic', 'tanh', 'relu'))
learning_rate_init = ht.ContinuousParameter('learning_rate_init', lower_bound=10**-5, upper_bound=0.1)
max_iter = ht.DiscreteParameter('max_iter', lower_bound=500, upper_bound=10**3)

hl1 = ht.DiscreteParameter('', lower_bound=50, upper_bound=250)
hl2 = ht.DiscreteParameter('', lower_bound=100, upper_bound=250)
hl3 = ht.DiscreteParameter('', lower_bound=1, upper_bound=100)
hidden_layer_sizes = ht.TupleParameter('hidden_layer_sizes', values=(hl1, hl2, hl3))

hypers = [activation, learning_rate_init, max_iter, learning_rate, hidden_layer_sizes]

# define a Hypertune object
tuner = ht.HyperTune(algorithm=MLPRegressor,
					parameters=hypers, 
					train_func=MLPRegressor.fit,
					objective_func=MLPRegressor.score,
					train_func_args=(X_train, y_train), 
					objective_func_args=(X_test, y_test),
					max_evals=10**2,
					maximize=True,
					num_replications=30)

# tune and print results
results = tuner.tune()
print(results)
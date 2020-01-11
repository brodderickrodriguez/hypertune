import torch
import torch.nn as nn
import hypertune as ht


class Net:
	def __init__(self, torch_optimizer, eta, max_iter, 
				hidden_layer_sizes, topology):
		self.torch_optimizer = torch_optimizer
		self.eta = eta
		self.max_iter = max_iter
		self.hidden_layer_sizes = hidden_layer_sizes
		self.topology = topology

	def predict(self, X):
		return self.model_(X)

	def _build_model(self, in_dim, out_dim):
		t = (in_dim, *self.hidden_layer_sizes, out_dim)
		layers, a, b = [], t[0], t[1]

		for i in range(len(self.hidden_layer_sizes) + 1):
			layers.append(self.topology[i](a, b))
			if i < len(self.hidden_layer_sizes):
				layers.append(nn.ReLU())
				a, b = b, t[i + 2]

		self.model_ = nn.Sequential(*tuple(layers))

	def fit(self, X, y):
		self._build_model(X.shape[1], y.shape[1])
		self._optimizer_ = self.torch_optimizer(params=self.model_.parameters(), lr=self.eta)
		self._loss_func_ = nn.MSELoss(reduction='sum')

		for t in range(self.max_iter):
			y_hat = self.predict(X)
			loss = self._loss_func_(y, y_hat)
			self._optimizer_.zero_grad()
			loss.backward()
			self._optimizer_.step()
			
	def mse(self, X, y):
		y_hat = self.predict(X)
		return self._loss_func_(y, y_hat).item()


# make an example dataset
p = 750
X = np.random.rand(10**4, 3)
y = np.random.rand(10**4)
X_train, X_test, y_train, y_test = X[:p], X[p:], y[:p], y[p:]

# define the target hyperparameters
too = torch.optim.Adam, torch.optim.Adadelta, torch.optim.Adagrad, torch.optim.ASGD
to = ht.CategoricalParameter('torch_optimizer', options=too)
eta = ht.ContinuousParameter('eta', lower_bound=1e-10, upper_bound=1e-1)
mi = ht.DiscreteParameter('max_iter', lower_bound=1e2, upper_bound=1e4)
hl1 = ht.DiscreteParameter('', lower_bound=10, upper_bound=100)
hl2 = ht.DiscreteParameter('', lower_bound=10, upper_bound=100)
hls = ht.TupleParameter('hidden_layer_sizes', values=(hl1, hl2))
tp1 = ht.CategoricalParameter('', options=(nn.Linear,))
tp2 = ht.CategoricalParameter('', options=(nn.Linear,))
tp3 = ht.CategoricalParameter('', options=(nn.Linear,))
top = ht.TupleParameter('topology', values=(tp1, tp2, tp3))
hypers = [to, eta, mi, hls, top]

# define a Hypertune object
tuner = ht.HyperTune(algorithm=Net,
					parameters=hypers,
					train_func=Net.fit,
					objective_func=Net.mse,
					train_func_args=(X_train, y_train), 
					objective_func_args=(X_test, y_test),
					max_evals=100,
					maximize=False,
					num_replications=10)

# tune and print results
results = tuner.tune()
print(results)

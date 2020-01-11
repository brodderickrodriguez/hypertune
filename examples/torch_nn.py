# Brodderick Rodriguez
# 12 Nov 2019

import torch
import torch.nn as nn
import hypertune as ht
import datasets 


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
		assert len(self.hidden_layer_sizes) + 1 == len(self.topology), \
				'hidden_layer_sizes and topology dims must match'

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

		self._optimizer_ = self.torch_optimizer(params=self.model_.parameters(),
												lr=self.eta)

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


def get_data():
	N, D_in, D_out = 100, 30, 3
	X = torch.randn(N, D_in)
	y = torch.randn(N, D_out)
	return X, y


def simple_run():
	X, y = get_data()

	net = Net(torch_optimizer=torch.optim.Adam, 
				eta=1e-3, 
				max_iter=5000,
				hidden_layer_sizes=(100,), 
				topology=(nn.Linear, nn.Linear))

	net.fit(X, y)
	print(net.mse(X, y))


def tune():
	X, y = get_data()

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

	tuner = ht.HyperTune(algorithm=Net,
						parameters=hypers,
						train_func=Net.fit,
						objective_func=Net.mse,
						train_func_args=(X, y), 
						objective_func_args=(X, y),
						max_evals=100,
						maximize=False,
						num_replications=1)

	tuner.tune()
	print(tuner.get_results())
	


# simple_run()
tune()


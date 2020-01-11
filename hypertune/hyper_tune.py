# Brodderick Rodriguez
# 08 Nov 2019

import numpy as np
import warnings
import multiprocessing
from . import optimizers
from .parameter import Parameter
from . import util


class HyperTune:
	def __init__(self, algorithm, parameters, train_func, objective_func, 
				train_func_args=None, 
				objective_func_args=None,
				max_evals=100, 
				optimizer=optimizers.PSO(),
				maximize=True,
				num_replications=1):

		self.algorithm = algorithm
		self.parameters = parameters
		self._train_func = train_func
		self._objec_func = objective_func
		self._train_objec_func_args = train_func_args, objective_func_args
		self.num_replications = num_replications

		self.optimizer = optimizer
		self.max_evals = max_evals
		self.maximize = maximize
		self.default_value = np.NINF if self.maximize else np.inf

		# self._check_arguments()

		# assert isinstance(optimizer, optimizers.Optimizer), \
		# 				'{} is not of type Optimizer'.format(optimizer)

		# assert num_replications >= 1, 'num_replications must be >= 1'

		# try:
		# 	util.get_kwarg_params(self.algorithm, parameters)
		# except AttributeError:
		# 	raise ValueError('all hyperparameters must be of ' \
		# 					'type hypertune.Parameter')

	def _check_arguments(self):
		tuple_e_msg = 'function args must be a tuple'
		train_func_args, objec_func_args = self._train_objec_func_args
		assert isinstance(train_func_args, tuple), tuple_e_msg
		assert isinstance(objec_func_args, tuple), tuple_e_msg

		kwarg_e_msg = 'all hyperparameters must be of type hypertune.Parameter'
		try:
			util.get_kwarg_params(self.algorithm, self.parameters)
		except AttributeError:
			raise ValueError(kwarg_e_msg)

		optimizer_e_msg = '{} is not of type Optimizer'.format(self.optimizer)
		assert isinstance(self.optimizer, optimizers.Optimizer), optimizer_e_msg

		assert self.num_replications >= 1, 'num_replications must be >= 1'

	@staticmethod
	def _process_evaluator(a, kwargs, train, objec, train_objec_func_args, 
							num_replications, ret_d, i):
		train_func_args, objec_func_args = train_objec_func_args
		results = np.zeros(num_replications)

		def _call(func, algo, args):
			try:
				return func(algo, *args)
			except TypeError:
				return func(algo)

		for j in range(num_replications):
			algo = a(**kwargs)
			_call(train, algo, train_func_args)
			result_j = _call(objec, algo, objec_func_args)
			results[j] = result_j

		with warnings.catch_warnings():
			warnings.simplefilter('ignore', category=RuntimeWarning)
			result = np.nanmean(results)

		ret_d[i] = result

	def evaluator(self, raw_hyper_params):
		manager = multiprocessing.Manager()
		ret_d = manager.dict()
		processes = []

		results = np.zeros(raw_hyper_params.shape[0])
		results.fill(self.default_value)

		for i, raw_hyper_param in enumerate(raw_hyper_params):
			kwargs = util.get_kwarg_values_init(self.algorithm.__init__, 
												self.parameters, 
												raw_hyper_param)

			process_args = (self.algorithm, kwargs, self._train_func, 
							self._objec_func, self._train_objec_func_args, 
							self.num_replications, ret_d, i)

			p = multiprocessing.Process(target=self._process_evaluator, 
										args=process_args)
			processes.append(p)
			p.start()

		for i, p in enumerate(processes):
			p.join()

			if i in ret_d and not np.isnan(ret_d[i]):
				results[i] = ret_d[i]

		print(results)
		return results

	def tune(self):
		self.optimizer.sender = self
		self.optimizer.state_shape = sum([p.shape for p in self.parameters])
		self.optimizer.optimize()
		self._optimized_results_ = self.optimizer.get_results()

		raw_params = self._optimized_results_['raw_params']
		kwargs = util.get_kwarg_values_noninit(self.algorithm, 
												self.parameters, 
												raw_params)

		self._optimized_results_['readable'] = kwargs
		return self.get_results()
	
	def get_parameter_values(self):
		return self._optimized_results_['readable']

	def get_fitness(self):
		return self._optimized_results_['fitness']

	def get_results(self):
		return {'objective_fn_value': self.get_fitness(),
				'params': self.get_parameter_values()}

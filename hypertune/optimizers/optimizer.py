# Brodderick Rodriguez
# 12 Nov 2019


class Optimizer:
	def __init__(self, sender=None):
		self.sender = None
		self.state_shape = 0

	def is_better(self, a, b):
		"""
		returns a > b if PSO is maximizing, otherwise returns a <= b
		"""
		r = a > b
		return r if self.sender.maximize else not r

	def optimize(self, *args, **kwargs):
		raise NotImplementedError

	def get_results(self, *args, **kwargs):
		d = {'fitness': None, 'raw_params': None}
		return d

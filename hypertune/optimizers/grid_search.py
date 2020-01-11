# Brodderick Rodriguez
# 18 Nov 2019

import numpy as np
from .optimizer import Optimizer


class GridSearch(Optimizer):	
	def __init__(self, depth=1, resolution=0.1, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def optimize(self):
		raise NotImplementedError

	def get_results(self):
		d = {'fitness': None, 'raw_params': None}
		return d

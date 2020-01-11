# Brodderick Rodriguez
# 13 Nov 2019

import numpy as np
import unittest
import hypertune as ht

class A:
	def __init__(self, a, b=None):
		self.a, self.b = a, b

	def fit(self, one, two):
		pass

	def acc(self, one, two):
		pass


class GridSearchTest(unittest.TestCase):
	def test_import(self):
		print('go')


		a = ht.ContinuousParameter('a', lower_bound=0, upper_bound=1)
		hypers = [a]


		gs = ht.optimizers.GridSearch(depth=1, resolution=0.1)
		tuner = ht.HyperTune(algorithm=A,
							parameters=hypers,
							optimizer=gs,
							train_func=A.fit,
							objective_func=A.acc,
							max_evals=100,
							maximize=False,
							num_replications=1)

		results = tuner.tune()
		print(results)

if __name__ == '__main__':
	unittest.main()

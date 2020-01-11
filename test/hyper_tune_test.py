# Brodderick Rodriguez
# 13 Nov 2019

import numpy as np
import unittest
import hypertune as ht


def fit(*args, **kwargs): pass

def acc(*args, **kwargs): return np.random.rand()


class CC: 
	def __init__(self, a, b): 
		self.a, self.b = a, b
		assert a == 'cc.a'
		assert b == 'cc.b'


class AI:
	def __init__(self, O=None, a=None): 
		self.O, self.a = O, a
		assert a == 'ai.a'


class CD:
	def __init__(self, a, b, O):
		self.a, self.b, self.O = a, b, O
		assert a == 'cd.a'
		assert b == 'cd.b'


class CE:
	def __init__(self, a=None, b=None, c=None): 
		self.a, self.b, self.c = a, b, c


class HypertuneTest(unittest.TestCase):
	def test_default_parameters(self):
		cea = ht.ConstantParameter('a', value='ce.a')
		cec = ht.ConstantParameter('c', value='ce.c')

		toa = (None,) * 2
		args = CE, [cec, cea], fit, acc, *toa, 0
		print(args)

		r = ht.HyperTune(*args).tune()['params']
		exp = {'a': 'ce.a', 'c': 'ce.c'}
		self.assertEqual(r, exp)

	def test_parameter_scope(self):
		cca = ht.ConstantParameter('a', value='cc.a')
		ccb = ht.ConstantParameter('b', value='cc.b')
		aio = ht.ObjectParameter('O', obj=CC, parameters=(cca, ccb))
		aia = ht.ConstantParameter('a', value='ai.a')

		r = ht.HyperTune(AI, [aia, aio], fit, acc, max_evals=0).tune()['params']
		exp = {'O': {'a': 'cc.a', 'b': 'cc.b'}, 'a': 'ai.a'}
		self.assertEqual(r, exp)

		r = ht.HyperTune(AI, [aia, aio], fit, acc, max_evals=0).tune()['params']
		self.assertEqual(r, exp)

		aio = ht.ObjectParameter('O', obj=CC, parameters=(ccb, cca))
		r = ht.HyperTune(AI, [aia, aio], fit, acc, max_evals=0).tune()['params']
		self.assertEqual(r, exp)

	def test_parameter_scope2(self):
		cca = ht.ConstantParameter('a', value='cc.a')
		ccb = ht.ConstantParameter('b', value='cc.b')
		aio = ht.ObjectParameter('O', obj=CC, parameters=(ccb, cca))
		aia = ht.ConstantParameter('a', value='ai.a')

		cdo = ht.ObjectParameter('O', obj=AI, parameters=(aio, aia))
		cda = ht.ConstantParameter('a', value='cd.a')
		cdb = ht.ConstantParameter('b', value='cd.b')
		
		r = ht.HyperTune(CD, [cda, cdb, cdo], fit, acc, max_evals=0).tune()['params']
		exp = {'a': 'cd.a', 'b': 'cd.b', 'O': {'O': {'a': 'cc.a', 'b': 'cc.b'}, 'a': 'ai.a'}}
		self.assertEqual(r, exp)

if __name__ == '__main__':
	unittest.main()

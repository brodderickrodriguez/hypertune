# Brodderick Rodriguez
# 13 Nov 2019

import numpy as np
import unittest
import hypertune as ht


def rand(size=1):
	return np.random.rand(size) * 2 - 1


def DA(): pass


def DB(a): return a


class CA:
	def __init__(self): pass


class CB:
	def __init__(self, a): self.a = a


class CC: 
	def __init__(self, a, b): self.a, self.b = a, b


class ParameterTest(unittest.TestCase):
	def test_constant_param(self):
		c = ht.ConstantParameter('c', value=10)		
		self.assertEqual(c.shape, 0)
		self.assertEqual(c.name, 'c')

		r = rand()
		v = c.get_val(r)
		self.assertEqual(v, 10)

		ca = CA()
		c = ht.ConstantParameter('a', value=ca)
		v = c.get_val([-1])
		self.assertEqual(v, ca)

		da = DA()
		c = ht.ConstantParameter('da', value=da)
		v = c.get_val([-1])
		self.assertEqual(v, da)

	def test_continuous_param(self):
		c = ht.ContinuousParameter('c', lower_bound=0, upper_bound=0)
		self.assertEqual(c.shape, 1)

		r = rand()
		v = c.get_val(r)
		self.assertEqual(v, 0)

		c = ht.ContinuousParameter('c', lower_bound=0, upper_bound=1)
		r = rand()
		v = c.get_val(r)
		exp = (r[0] + 1) / 2
		self.assertEqual(v, exp)

		lb, ub = -100, 500
		c = ht.ContinuousParameter('c', lower_bound=lb, upper_bound=ub)
		r = rand()
		v = c.get_val(r)
		exp = ((r[0] + 1) / 2) * (ub - lb) + lb
		self.assertEqual(v, exp)

	def test_discrete_param(self):
		d = ht.DiscreteParameter('d', lower_bound=0, upper_bound=1)
		self.assertEqual(d.name, 'd')
		self.assertEqual(d.shape, 1)

		r = rand()
		v = d.get_val(r)
		exp = int(np.round((r[0] + 1) / 2))
		self.assertEqual(v, exp)

	def test_tuple_param(self):
		t = ht.TupleParameter('t', values=(1, 1, 1))
		self.assertEqual(t.shape, 0)

		c = ht.ContinuousParameter('c', lower_bound=0, upper_bound=1)
		d = ht.DiscreteParameter('d', lower_bound=0, upper_bound=1)
		co = ht.ConstantParameter('co', value='world')
		t = ht.TupleParameter('t', values=(c, d, co))
		self.assertEqual(t.shape, 2)

		r = [-1, 0.5]
		v = t.get_val(r)
		exp = (0.0, 1, 'world')
		self.assertEqual(v, exp)

		t = ht.TupleParameter('t', values=(c, {'z': -1}, d))
		r = [-1, 0.5]
		v = t.get_val(r)
		exp = (0.0, {'z': -1}, 1)
		self.assertEqual(v, exp)

	def test_categorical_param(self):
		c = ht.CategoricalParameter('c', options=('a', 'b'))
		self.assertEqual(c.name, 'c')
		self.assertEqual(c.shape, 1)

		v = c.get_val([-1])
		self.assertEqual(v, 'a')

		co = ht.ContinuousParameter('co', lower_bound=0, upper_bound=1)
		con = ht.ConstantParameter('con', value=100)
		c = ht.CategoricalParameter('c', options=('a', co, con))
		self.assertEqual(c.shape, 2)

	def test_object_param(self):
		o = ht.ObjectParameter('o', obj=CA, parameters=tuple([]))
		self.assertEqual(o.shape, 0)

		prams = 'o', DA, (1,)
		self.assertRaises(ValueError, ht.ObjectParameter, *prams)

		cba = ht.ConstantParameter('a', value='a')
		o = ht.ObjectParameter('o', obj=CB, parameters=(cba,))
		self.assertEqual(o.shape, 0)

		dba = ht.ConstantParameter('a', value='aaa')
		o = ht.ObjectParameter('', obj=DB, parameters=(dba,))
		v = o.get_val([1])
		self.assertEqual(v, 'aaa')

		cba = ht.DiscreteParameter('a', lower_bound=0, upper_bound=1)
		o = ht.ObjectParameter('o', obj=CB, parameters=(cba,))
		self.assertEqual(o.shape, 1)

		p = [-1]
		v = o.get_val(p)
		self.assertEqual(v.a, cba.get_val(p))

		cca = ht.DiscreteParameter('a', lower_bound=0, upper_bound=1)
		ccb = ht.DiscreteParameter('b', lower_bound=-100, upper_bound=50)
		o = ht.ObjectParameter('o', obj=CC, parameters=(cca, ccb))
		p = [0.5, 0.75]
		v = o.get_val(p)
		self.assertEqual(v.a, cca.get_val(p[:1]))
		self.assertEqual(v.b, ccb.get_val(p[1:]))

		cca = ht.ConstantParameter('a', value='aaa')
		ccb = ht.ConstantParameter('b', value='bbb')
		o = ht.ObjectParameter('o', obj=CC, parameters=(ccb, cca))
		p = [0.5, 0.75]
		v = o.get_val(p)
		self.assertEqual(v.a, cca.get_val(p[:1]))
		self.assertEqual(v.b, ccb.get_val(p[1:]))


if __name__ == '__main__':
	unittest.main()

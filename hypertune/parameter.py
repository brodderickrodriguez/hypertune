# Brodderick Rodriguez
# 08 Nov 2019

import numpy as np
from . import util


class Parameter:
	def __init__(self, name):
		assert isinstance(name, str), 'name must be a string'
		self.name = name
		self.shape = 0

	def __str__(self):
		return '{} {}'.format(type(self).__name__, self.__dict__)

	def __repr__(self):
		return self.__str__()

	def get_val(self, x):
		raise NotImplementedError

	def get_dict(self, x):
		return self.get_val(x)


class ConstantParameter(Parameter):
	def __init__(self, name, value):
		super().__init__(name)
		self.value = value

	def get_val(self, x):
		return self.value


class ContinuousParameter(Parameter):
	def __init__(self, name, lower_bound, upper_bound):
		super().__init__(name)
		bounds_error_msg = 'bounds must be a real number'
		assert isinstance(lower_bound, int) or \
				isinstance(lower_bound, float), bounds_error_msg
		assert isinstance(upper_bound, int) or \
				isinstance(upper_bound, float), bounds_error_msg
		assert lower_bound <= upper_bound, \
				'lower_bound must be less than or equal to upper_bound'
		self.lb = lower_bound
		self.ub = upper_bound
		self.shape = 1

	def get_val(self, x):
		_x = (x[0] + 1) / 2
		_x = _x * (self.ub - self.lb) + self.lb
		return _x


class DiscreteParameter(ContinuousParameter):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs) 

	def get_val(self, x):
		_x = super().get_val(x)
		_x = int(np.round(_x))
		return _x


class TupleParameter(Parameter):
	def __init__(self, name, values):
		super().__init__(name) 
		self.values = values
		self.shape = sum([vi.shape for vi in values 
						if isinstance(vi, Parameter)])

	def _get_param(self, x, get_func):
		_x, offset = [], 0
		for v in self.values:
			if isinstance(v, Parameter):
				vx = x[offset: offset + v.shape]
				offset += v.shape
				get = getattr(v, get_func)
				_x.append(get(vx))
			else:
				_x.append(v)
		return tuple(_x)

	def get_val(self, x):
		return self._get_param(x, get_func='get_val')

	def get_dict(self, x):
		return self._get_param(x, get_func='get_dict')


class CategoricalParameter(TupleParameter):
	def __init__(self, name, options):
		super().__init__(name, options)
		self.options = options
		self.shape += 1

	def get_val(self, x):
		values = super().get_val(x[1:])
		normalized_idx = ((x[0] + 1) / 2) * (len(self.options) - 1)
		idx = int(np.round(normalized_idx))
		return values[idx]

	def get_dict(self, x):
		d = super().get_dict(x[1:])
		normalized_idx = ((x[0] + 1) / 2) * (len(self.options) - 1)
		idx = int(np.round(normalized_idx))
		return d[idx]

class ObjectParameter(TupleParameter):
	def __init__(self, name, obj, parameters):
		super().__init__(name, parameters)
		assert callable(obj), 'obj must be callable'
		self._check_parameters(parameters)
		self.obj = obj

	def _check_parameters(self, parameters):
		for p in parameters:
			if not isinstance(p, Parameter):
				raise ValueError('parameters for object must be ' \
								'of type Parameter')

	def get_val(self, x):
		kwargs = util.get_kwarg_values_init(self.obj, self.values, x)
		return self.obj(**kwargs)

	def get_dict(self, x):
		vals = util.get_kwarg_values_noninit(self.obj, self.values, x)
		return vals

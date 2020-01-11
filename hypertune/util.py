# Brodderick Rodriguez
# 13 Nov 2019

import inspect
import hypertune as ht


def _get_callable_params(cble):
	params_names = inspect.getfullargspec(cble).args

	if 'self' in params_names: 
		params_names.remove('self')

	return params_names


def get_kwarg_params(cble, params):
	p_lookup = {p.name: p for p in params}
	c_names = _get_callable_params(cble)
	c_names = [cpn for cpn in c_names if cpn in p_lookup]
	kwarg_params = {c: p_lookup[c] for c in c_names}
	return kwarg_params


def get_kwarg_values(kwarg_params, x, get_func):
	kwarg_values, offset = {}, 0

	for kw, param in kwarg_params.items():
		xp = x[offset: param.shape + offset]
		offset += param.shape
		get = getattr(param, get_func)
		kwarg_values[kw] = get(xp)

	return kwarg_values


def get_kwarg_values_init(cble, params, x):
	kwarg_params = get_kwarg_params(cble, params)
	return get_kwarg_values(kwarg_params, x, get_func='get_val')


def get_kwarg_values_noninit(cble, params, x):
	kwarg_params = get_kwarg_params(cble, params)
	return get_kwarg_values(kwarg_params, x, get_func='get_dict')


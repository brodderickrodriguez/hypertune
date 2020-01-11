# Brodderick Rodriguez
# 10 Nov 2019

from sklearn.model_selection import train_test_split
import sklearn.datasets as skds
import numpy as np


def normalize(z):
	_max = np.max(z)
	_min = np.min(z)
	z_ = (z - _min) / (_max - _min)
	return z_

def split(X, y, test_size=0.33):
	return train_test_split(X, y, test_size=test_size)


def _get_dataset(ds, return_splits, test_size):
	X, y = ds(return_X_y=True)
	
	if not return_splits:
		return X, y

	return split(X, y, test_size)


# classification
def iris(return_splits=True, test_size=0.33):
	return _get_dataset(skds.load_iris, return_splits, test_size)


# classification
def faces(return_splits=True, test_size=0.33):
	return _get_dataset(skds.fetch_lfw_people, return_splits, test_size)


# classification
def digits(return_splits=True, test_size=0.33):
	return _get_dataset(skds.load_digits, return_splits, test_size)


# regression
def housing(return_splits=True, test_size=0.33):
	X, y = _get_dataset(skds.fetch_california_housing, False, test_size)
	y = y.astype('int')

	if not return_splits:
		return X, y

	return split(X, y, test_size)


# regression
def diabetes(return_splits=True, test_size=0.33):
	return _get_dataset(skds.load_diabetes, return_splits, test_size)

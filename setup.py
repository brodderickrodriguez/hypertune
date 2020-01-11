# Brodderick Rodriguez
# 08 Nov 2019

from setuptools import setup


setup(name='hypertune',
      version='0.1.0',
      description='A package to tune ML hyperparameters using Particle Swarm Optimization.',
      url='https://github.com/brodderickrodriguez/pso_hyper_tune',
      author='Brodderick Rodriguez',
      author_email='bcr@brodderick.com',
      license='Apache-2.0',
      packages=['hypertune', 'hypertune/optimizers'],
      zip_safe=False)

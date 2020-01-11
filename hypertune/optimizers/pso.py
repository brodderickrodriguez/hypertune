# Brodderick Rodriguez
# 08 Nov 2019

import numpy as np
from enum import Enum
from .optimizer import Optimizer


class _Particle:
	def __init__(self, shape, x_val_init):
		# weights for self, others
		self.phi = [2.05, 2.05]

		# set current vector position to something in range [-1.0, 1.0]
		self.x = np.random.rand(shape) * 2 - 1

		# set the value of its current location
		self.x_val = x_val_init

		# set velocity vector to zeros
		self.v = np.zeros(shape)

		# set the best_x seen as the current x vector
		self.best_x = self.x

		# set the best_x value as the current x value
		self.best_x_val = self.x_val

		# other is either the global (entire pop.) or the neighborhood
		# set the other best_x as the current best_x for this particle
		self.other_best_x = self.best_x

		# other refers to either the global (entire pop.) or the neighborhood
		# set the other best_x value as the current best_x vector
		self.other_best_x_val = self.best_x_val

		# other refers to either the global (entire pop.) or the neighborhood
		# depending on which topology you are using
		self.others = []

	def move(self, k):
		# The parameter, κ, in equation (16.32) controls the exploration and 
		# exploitation abilities of the swarm. For κ ≈ 0, fast convergence is 
		# obtained with local exploitation. The swarm exhibits an almost 
		# hill-climbing behavior. On the other hand, κ ≈ 1 results in slow 
		# convergence with a high degree of exploration.

		# phi = phi_1 + phi2
		phi = np.sum(self.phi)

		# constriction coefficient
		chi = (2 * k) / np.abs(2 - phi - np.sqrt(phi * (phi - 4)))

		# the v update for the self component
		self_component = self.phi[0] * (self.best_x - self.x)

		# the v update for the others component
		others_component = self.phi[1] * (self.other_best_x - self.x)

		# the complete update using constriction coefficient
		update = chi * (self.v + self_component + others_component)

		# finally update velocity and location
		self.v = update
		self.x += self.v

		# constrict self.x to be in the range [-1.0, 1.0]
		self.x[self.x < -1.0] = -1.0
		self.x[self.x > 1.0] = 1.0

	def __str__(self):
		return 'Particle: x_val: {}'.format(self.x_val)

	def __repr__(self):
		return self.__str__()


class _Topology(Enum):
	@staticmethod
	def _ring(pso, ns=3):
		assert ns <= len(pso._population), 'population must be >= 3'
		for particle in pso._population:
			others = np.random.choice(pso.population, ns, replace=False)

			if particle not in others:
				others[0] = particle

			particle.others = list(others)

	@staticmethod
	def _star(pso):
		for particle in pso._population:
			particle.others = pso._population

	Star = _star
	Ring = _ring


class PSO(Optimizer):
	Topology = _Topology
	
	def __init__(self, population_size=10, 
				topology=Topology.Star, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self._topology_setup = topology
		self.population_size = population_size

		# k has something to do with constriction coeff
		self._k = 0.5

	def __str__(self):
		return ''.join([str(p) + '\n' for p in sorted(self._population)])

	def _sorted(self, particles, attr):
		k = lambda p: getattr(p, attr)
		s = sorted(particles, key=k, reverse=self.sender.maximize)
		return s

	def optimize(self):
		self._population = [_Particle(self.state_shape, self.sender.default_value) 
							for _ in range(self.population_size)]

		evals = (self.sender.max_evals // len(self._population)) + 1
		print(self.sender.max_evals)

		self._topology_setup(self)

		for _ in range(self.sender.max_evals):
			self._evolutionary_cycle()	

	def _evaluate_population(self):
		particles_x = np.zeros((len(self._population), self.state_shape))

		for i, particle in enumerate(self._population):
			particles_x[i] = particle.x

		fitnesses = self.sender.evaluator(particles_x)

		for i, particle in enumerate(self._population):
				particle.x_val = fitnesses[i]

	def _evolutionary_cycle(self):
		self._evaluate_population()

		for particle in self._population:
			if self.is_better(particle.x_val, particle.best_x_val):
				particle.best_x = particle.x
				particle.best_x_val = particle.x_val

		for particle in self._population:
			best_other = self._sorted(particle.others, attr='x_val')[0]

			if self.is_better(best_other.best_x_val, particle.other_best_x_val):
				particle.other_best_x_val = best_other.best_x_val
				particle.other_best_x = best_other.best_x

		for particle in self._population:
			particle.move(k=self._k)

	def get_results(self):
		best = self._sorted(self._population, attr='best_x_val')[0]
		return {'fitness': best.best_x_val, 'raw_params': best.best_x}

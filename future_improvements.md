## Improvements

Possible improvements:
* Add more Optimizers
* Vectorize PSO Particle into np.array
* add support for evolving the depth of a NN through extension of TupleParameter
* add a logger with the ability to supress warnings originating from the target algorithm
* allow train, objective func to take dicts or tuples
* improve how HyperTune.max_evals works
* add ability for passing hypertune params to non hypertune object which is encapsulated in a hypertune object i.e. Param(Class(Param(), Param()))
* Finish examples
	* update About HyperTune Parameters notebook
	* update neuroevolution example
* add gridsearch optimizer
	* need to first add ability to cap number of processes. This way grid_search can send all possible solutions to HyperTune object. 
	* Need to think about a solution for converting [-1.0 1.0] to all available options for parameter. Otherwise this will cause repetative target algorithm instances. Think about adding to Parameter classes a way to enumberate all possible valules. 


Clean up:
* clean up Tuple, Cat, Object param get_val, get_dict functions

# Bayesian nonparametric modulated renewal processes

See notebooks in `tutorial/` for a basic usage overview.
See `validation/` for model validation code.
Inspired by probabilistic programming library [Pyro](https://github.com/pyro-ppl/pyro).


## Motivation

Many powerful statistical models have been proposed in the literature for analyzing neural data, 
but a unified library for many of these models is lacking. Models may be written in different 
programming languages, and lack support for modern hardware and parallelization. This library 
provides a general framework based on probabilistic programming (generalized variational inference) 
and supports scalability of deep learning via PyTorch.


## Dependencies:
#. JAX
#. NumPy
#. SciPy
#. tqdm

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
#. [JAX](https://jax.readthedocs.io/en/latest/#)
#. [NumPy](https://numpy.org/)
#. [SciPy](https://scipy.org/)
#. [tqdm](https://github.com/tqdm/tqdm)

Formatting
#. [ufmt](https://pypi.org/project/ufmt/)

Code analysis linter
#. [pylint](https://www.pylint.org/)



- fit GP-GLM
- fit modulated renewal
- schematic plots
- validation
- biophysical
- place cells


- ELBO and fitting
- Sample STGP
- pathwise conditioned KS test
- neural correlations
- LVM
# NeuroPPL: Probabilistic programming for neuroscience

See notebooks in `tutorial/` for a basic usage overview.
See `validation/` for model validation code.
Inspired by probabilistic programming library [Pyro](https://github.com/pyro-ppl/pyro).


## Motivation

Many powerful statistical models have been proposed in the literature for analyzing neural data, 
but a unified library for many of these models is lacking. Models may be written in different 
programming languages, and lack support for modern hardware and parallelization. This library 
provides a general framework based on probabilistic programming (generalized variational inference) 
and supports scalability of deep learning via PyTorch.



## Primitives

There are three kinds of objects that form the building blocks:
1. Input group *p(X,Z)* and *q(Z)*
2. Mapping *p(F|X,Z)*
3. Likelihood *p(Y|F)*

The overal generative model is specified along with the variational posterior through 
these primitives. Input groups can contain observed and latent variables, with priors 
on the latent variables that can be another mapping itself. This allows construction 
of hierarchical models.


## Models implemented

* Linear-nonlinear and GPs
* RNNs
* NFs
* LVMs
    - Toroidal and spherical latent spaces ([Manifold GPLVM](https://arxiv.org/abs/2006.07429))
    - GP priors (deep GPs when going beyond two GP layers)
    - AR(1) temporal prior on latents
    - HMM latent structure (discrete latent)
    - Flow-based posteriors (TODO)
* GLM filters
    - spike-history couplings
    - spike-spike couplings
    - stimulus history
* Inhomogenenous renewal point processes
    - Gamma
    - Inverse Gaussian
    - Log Normal
* Count process likelihoods
    - Poisson
    - Zero-inflated Poisson
    - Negative binomial
    - Conway-Maxwell-Poisson
* Gaussian likelihoods
    - Univariate
    - Multivariate (TODO)


## Optimization procedures

* PyTorch built-in optimizers (LBFGS special case)
* Natural gradient
* Newton's method
* HMC/NUTS/MCMC (TODO)


## Dependencies:
#. PyTorch
#. NumPy
#. SciPy
#. tqdm


## Contributors:
David Liu (main code)
Mattijs De Paepe (debugging regression models, circular correlation statistics in stats.py)
Kristopher T. Jensen (debugging GPLVM code)
# Bayesian nonparametric (non-)renewal processes

See notebooks in `notebooks/` for a basic usage overview.
See `scripts/` for model fitting, analysis and plotting code.


## Motivation


TODO:
- Replot the hc3 related and BNPP plots with -1 margin
- log p values
- Inset of rate maps from 5D, with 3 ISIs on the side
- plot the tuning for baseline models, and ISIs as well

- Add dots to violin plots?
- Tuning of more models, all BNPPs and baselines?
- put library subset in and change path imports

- Try plot the tunings of baseline models too, e.g. rate and CV of those models... if time permits, use conditional with same past spike as the one used for that




## Dependencies:
#. [JAX](https://jax.readthedocs.io/en/latest/#)
#. [NumPy](https://numpy.org/)
#. [SciPy](https://scipy.org/)
#. [tqdm](https://github.com/tqdm/tqdm)

Formatting
#. [ufmt](https://pypi.org/project/ufmt/)

Code analysis linter
#. [pylint](https://www.pylint.org/)

import math

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.scipy.special import gammaln

from ..utils.jax import safe_log

_log_twopi = math.log(2 * math.pi)


# Gaussian
def Gaussian_log_likelihood(mu, obs_var, y):
    """
    Evaluate the log-Gaussian function log 𝓝(yₙ|fₙ,σ²).
    Can be used to evaluate Q approximation/cubature points.

    :param jnp.ndarray y: observed data yₙ (obs_dims,)
    :param jnp.ndarray f: mean, i.e. the latent function value fₙ (obs_dims,)
    :return:
        log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise (obs_dims,)
    """
    ll = -0.5 * (_log_twopi * safe_log(obs_var) + (y - mu) ** 2 / (obs_var + 1e-10))
    return ll


def sample_Gaussian_scalars(prng_state, mu, obs_var):
    return mu + jnp.sqrt(obs_var) * jr.normal(prng_state, shape=mu.shape)


# Poisson
def sample_Poisson_counts(prng_state, mean):
    return jr.poisson(prng_state, mean)


def Poisson_log_likelihood(lambd, y):
    ll = y * safe_log(lambd) - lambd - gammaln(y + 1)
    return ll


# ZIP
def sample_ZIP_counts(prng_state, mean, alpha):
    zero_mask = jr.bernoulli(prng_state, alpha)
    prng_state, _ = jr.split(prng_state)
    cnts = jr.poisson(prng_state, mean)
    return (1.0 - zero_mask) * cnts


def ZIP_log_likelihood(lambd, alpha, y):
    """
    Evaluate the Zero Inflated Poisson log-likelihood:
        log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
    for μ = g(fₙ), where g() is the link function

    :param y: observed data (yₙ) [scalar]
    :param f: latent function value (fₙ) [Q, 1]
    :return:
        log likelihood array (obs_dims,)
    """

    ll_n = (
        y * safe_log(lambd) - lambd - gammaln(y + 1.0) + safe_log(1.0 - alpha)
    )  # -log (1-alpha)*p(N)
    p = jnp.exp(ll_n)  # stable as ll < 0
    ll_0 = safe_log(alpha + p)  # probability of zero counts, stable for alpha = 0

    ll = lax.select(y == 0, ll_0, ll_n)
    return ll


# NB
def sample_NB_counts(prng_state, mean, r):
    s = (mean / r) * jr.gamma(
        prng_state, r
    )  # becomes delta around rate*tbin when r to infinity
    prng_state, _ = jr.split(prng_state)
    return jr.poisson(prng_state, s)


def NB_log_likelihood(lambd, r_inv, y):
    """
    The negative log likelihood function. Note that if disper_param is not None, it will use those values for
    the dispersion parameter rather than its own dispersion parameters.

    .. math::
        P(n|\lambda, r) = \frac{\lambda^n}{n!} \frac{\Gamma(r+n)}{\Gamma(r) \, (r+\lamba)^n} \left( 1 + \frac{\lambda}{r} \right)^{-r}

    where the mean is related to the conventional parameter :math:`\lambda = \frac{pr}{1-p}`

    For :math:`r \to \infty` we compute the likelihood as a correction to Poisson retaining :math:`\log{1 + O(r^{-1})}`.
    We parameterize the likelihood with :math:`r^{-1}`, as this allows one to reach the Poisson limit (:math:`r^{-1} \to 0`)
    using the series expansion.

    :param jnp.ndarray lambd: lambda parameter (mean) of shape (..., obs_dims)
    :param jnp.ndarray r_inv: 1/r
    :returns:
        log likelihood of shape (...), jnp.ndarray
    """
    asymptotic_mask = r_inv < 1e-3  # when r becomes very large
    r = 1.0 / (r_inv + asymptotic_mask)  # avoid NaNs

    fac_gammaln = gammaln(r + y) - gammaln(r)
    fac_power = -(y + r) * safe_log(r + lambd) + r * safe_log(r)

    ll_r = fac_power + fac_gammaln
    ll_r_inv = -lambd - jnp.log(
        1.0 + r_inv * (y**2 - y * (1 / 2 + lambd))
    )  # expand around small 1/r

    ll = y * safe_log(lambd) - gammaln(y + 1)

    # numerically stable for large range of r
    ll += lax.select(asymptotic_mask, ll_r_inv, ll_r)
    return ll


# CMP
def sample_CMP_counts(prng_state, lambd, nu):
    """
    Use rejection sampling to sample from the COM-Poisson count distribution. [1]

    References:

    [1] `Bayesian Inference, Model Selection and Likelihood Estimation using Fast Rejection
         Sampling: The Conway-Maxwell-Poisson Distribution`, Alan Benson, Nial Friel (2021)

    :param numpy.array rate: input rate of shape (..., time)
    :param float tbin: time bin size
    :param float eps: order of magnitude of P(N>1)/P(N<2) per dilated Bernoulli bin
    :param int max_count: maximum number of spike counts per bin possible
    :returns:
        inhomogeneous Poisson process sample (numpy.array)
    """
    mu = lambd ** (1 / nu)
    mu_ = jnp.floor(mu)

    def cond_fun(val):
        _, left_inds, _ = val
        return left_inds.any()

    def poiss_reject(val):
        Y, left_inds, prng_state = val
        prng_keys = jr.split(prng_state, 3)

        y_dash = jr.poisson(prng_keys[0], mu)  # sample Poisson

        log_alpha = (nu - 1) * (
            (y_dash - mu_) * safe_log(mu) - gammaln(y_dash + 1.0) + gammaln(mu_ + 1.0)
        )  # log acceptance

        u = jr.uniform(prng_keys[1], shape=mu.shape)
        alpha = jnp.exp(log_alpha)
        selected = (u <= alpha) * left_inds

        Y = jnp.where(selected, y_dash, Y)
        left_inds *= ~selected
        return Y, left_inds, prng_keys[2]

    def geom_reject(val):
        Y, left_inds, prng_state = val
        prng_keys = jr.split(prng_state, 3)

        p = 2 * nu / (2 * mu * nu + 1 + nu)
        u_0 = jr.uniform(prng_keys[0], shape=mu.shape)
        y_dash = jnp.floor(safe_log(u_0) / safe_log(1 - p))  # sample geom(p)

        a = jnp.floor(mu / (1 - p) ** (1 / nu))
        log_alpha = (a - y_dash) * safe_log(1 - p) + nu * (
            (y_dash - a) * safe_log(mu) - gammaln(y_dash + 1.0) + gammaln(a + 1.0)
        )  # log acceptance

        u = jr.uniform(prng_keys[1], shape=mu.shape)
        alpha = jnp.exp(log_alpha)
        selected = (u <= alpha) * left_inds

        Y = jnp.where(selected, y_dash, Y)
        left_inds *= ~selected
        return Y, left_inds, prng_keys[2]

    prng_states = jr.split(prng_state, 2)
    Y = jnp.empty_like(mu)
    Y, _, _ = lax.while_loop(
        cond_fun, poiss_reject, init_val=(Y, (nu >= 1), prng_states[0])
    )
    Y, _, _ = lax.while_loop(
        cond_fun, geom_reject, init_val=(Y, (nu < 1), prng_states[1])
    )

    return Y


def CMP_log_likelihood(lambd, nu, y, J):
    """
    :param jnp.ndarray f: input variables of shape (f_dims,)
    :returns:
        NLL of shape (obs_dims,)
    """
    powers = jnp.arange(J + 1)
    jfact = gammaln(powers + 1.0)

    log_lambda = safe_log(lambd)

    # partition function
    log_Z_term = powers[:, None] * log_lambda[None, :] - nu[None, :] * jfact[:, None]
    log_Z = jax.nn.logsumexp(log_Z_term, axis=0)

    ll = y * log_lambda - log_Z - nu * gammaln(y + 1)
    return ll


def CMP_moments(k, rate, nu, sim_time, J=100):
    """
    Compute central moments of CMP distribution

    :param np.array k: order of moment to compute
    :param np.array rate: input rate of shape (neurons, timesteps)
    """
    g = rate[None, ...] * sim_time
    log_g = jnp.log(jnp.maximum(g, 1e-12))
    nu = nu[None, ...]
    k = jnp.array([k])[:, None, None]  # turn into array

    n = jnp.arange(1, J + 1)[:, None, None]
    j = jnp.arange(J + 1)[:, None, None]
    lnum = log_g * j
    lden = gammaln(j + 1) * nu
    logsumexp_Z = jax.nn.logsumexp(lnum - lden, axis=-1)[None, ...]
    return jnp.exp(k * jnp.log(n) + log_g * n - logsumexp_Z - gammaln(n + 1) * nu).sum(
        0
    )


def sample_rate_rescaled_renewal(prng_state, ISI_sampler_func, ini_t_step, rates, dt):
    """
    :param jnp.ndarray rates: shape (obs_dims, ts)
    """
    obs_dims, ts = rates.shape

    def step(carry, inputs):
        rescaled_t_step, rescaled_t_spike = carry
        rate, prng_state = inputs

        spikes = jnp.where(
            rescaled_t_step >= rescaled_t_spike, jnp.ones(obs_dims), jnp.zeros(obs_dims)
        )
        rescaled_t_spike += jnp.where(
            rescaled_t_step >= rescaled_t_spike, ISI_sampler_func(prng_state), 0
        )

        rescaled_t_step += rate * dt
        return (rescaled_t_step, rescaled_t_spike), spikes

    init = (ini_t_step, ISI_sampler_func(prng_state))
    prng_states = jr.split(prng_state, ts)
    _, spikes = lax.scan(step, init=init, xs=(rates.T, prng_states))
    return spikes

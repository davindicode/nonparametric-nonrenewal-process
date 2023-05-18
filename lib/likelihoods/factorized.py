import math
from typing import Callable, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import block_diag, solve_triangular
from jax.scipy.special import gammaln

from ..utils.jax import mc_sample, safe_log, softplus, softplus_inv
from ..utils.linalg import gauss_hermite
from . import distributions

from .base import CountLikelihood, FactorizedLikelihood, LinkTypes

_log_twopi = math.log(2 * math.pi)


### density likelihoods ###
class Gaussian(FactorizedLikelihood):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    """

    pre_variance: jnp.ndarray

    def __init__(self, obs_dims, variance, array_type="float32"):
        """
        :param jnp.ndarray pre_variance: The observation noise variance, σ² (obs_dims,)
        """
        super().__init__(obs_dims, obs_dims, "none", array_type)
        self.pre_variance = softplus_inv(self._to_jax(variance))

    def log_likelihood(self, f, y):
        obs_var = softplus(self.pre_variance)
        return distributions.Gaussian_log_likelihood(f, obs_var, y)

    def variational_expectation(
        self,
        y,
        f_mean,
        f_cov,
        prng_state,
        jitter,
        approx_int_method,
        log_predictive=False,
    ):
        """
        Exact integration, overwrite approximate integration

        log Zₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]

        ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[log 𝓝(yₙ|fₙ,σ²)]

        :param jnp.array f_mean: q(f) mean with shape (f_dims,)
        :param jnp.array f_cov: q(f) mean with shape (f_dims, f_dims)
        """
        if log_predictive is False:
            obs_var = softplus(self.pre_variance)
            f_mean = f_mean[:, 0]
            f_var = jnp.diag(f_cov)  # diagonalize

            Ell = -0.5 * (
                _log_twopi
                + (f_var + (y - f_mean) ** 2) / (obs_var + 1e-10)
                + safe_log(obs_var)
            )  # (obs_dims,)

            Ell = jnp.nansum(Ell)  # sum over obs_dims
            return Ell

        else:
            return super().variational_expectation(
                y, f_mean, f_cov, prng_state, jitter, approx_int_method, log_predictive
            )

    def sample_Y(self, prng_state, f):
        """
        Gaussian

        :param jnp.ndarray f: input rate of shape (f_dims,)
        :returns:
            spike train of shape (obs_dims,)
        """
        obs_var = softplus(self.pre_variance)
        return distributions.sample_Gaussian_scalars(prng_state, f, obs_var)


#     """
#     Exact, ignore approx_int_func

#     log Zₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]

#     ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[log 𝓝(yₙ|fₙ,σ²)]

#     :param jnp.array f_mean: q(f) mean with shape (f_dims,)
#     :param jnp.array f_cov: q(f) mean with shape (f_dims, f_dims)
#     """
#         if derivatives:
#             # dE_dm: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. mₙ of q(f)
#             # dE_dV: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. Vₙ of q(f)
#             dEll_dm = (y - f_mean) / obs_var
#             dEll_dV = -0.5 / obs_var

#             if mask is not None:
#                 dEll_dm = jnp.where(mask, 0.0, dEll_dm)  # (obs_dims,)
#                 dEll_dV = jnp.where(mask, 0.0, dEll_dV)  # (obs_dims,)

#             dlambda_1 = (dEll_dm - 2 * dEll_dV * f_mean)[:, None]  # (f_dims, 1)
#             dlambda_2 = jnp.diag(dEll_dV)  # (f_dims, f_dims)

#         else:
#             dlambda_1, dlambda_2 = None, None


class PointProcess(FactorizedLikelihood):
    """
    The continuous-time point process with log intensity given by stochastic process

    Small time bin discrete approximation
    """

    dt: float

    def __init__(
        self,
        neurons,
        dt,
        link_type="log",
        array_type="float32",
    ):
        if link_type not in ["log", "softplus"]:
            raise ValueError("Link type must be log or softplus")

        super().__init__(neurons, neurons, link_type, array_type)
        self.dt = dt

    def log_likelihood(self, f, y):
        """
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ ϵ ℝ
        :return:
            log p(yₙ|fₙ), p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾
        """
        f = safe_log(self.inverse_link(f))
        return y * f - jnp.exp(f) * self.dt

    def variational_expectation(
        self,
        y,
        f_mean,
        f_cov,
        prng_state,
        jitter,
        approx_int_method,
        log_predictive=False,
    ):
        """
        Exact integration, overwrite approximate integration

        log Zₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]

        ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[log 𝓝(yₙ|fₙ,σ²)]

        :param jnp.array y: observations (obs_dims,)
        :param jnp.array f_mean: q(f) mean with shape (f_dims, 1)
        :param jnp.array f_cov: q(f) mean with shape (f_dims, f_dims)
        """
        if log_predictive is False and self.link_type == LinkTypes["log"]:  # exact
            f_mean = f_mean[:, 0]
            f_var = jnp.diag(f_cov)  # diagonalize

            Ell = y * f_mean - jnp.exp(f_mean + f_var / 2.0) * self.dt
            Ell = jnp.nansum(Ell)  # sum over obs_dims
            return Ell

        else:
            return super().variational_expectation(
                y, f_mean, f_cov, prng_state, jitter, approx_int_method, log_predictive
            )

    def sample_Y(self, prng_state, f):
        """
        Bernoulli process approximation with small dt

        :param jnp.ndarray f: input rate of shape (f_dims,)
        :returns:
            spike train of shape (obs_dims,)
        """
        if self.link_type == "softplus":
            rho = softplus(f)
        else:  # log
            rho = jnp.exp(f)

        rate = jnp.minimum(rho * self.dt, 1.0)
        return jr.bernoulli(prng_state, rate).astype(self.array_dtype())


### discrete likelihoods ###
class Bernoulli(FactorizedLikelihood):
    """
    Bernoulli likelihood is p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾, where P = E[yₙ=1|fₙ].
    Link function maps latent GP to [0,1].

    The error function likelihood = probit = Bernoulli likelihood with probit link.
    The logistic likelihood = logit = Bernoulli likelihood with logit link.

    The logit link function:
        P = E[yₙ=1|fₙ] = 1 / 1 + exp(-fₙ)

        The Probit link function, i.e. the Error Function Likelihood:
        i.e. the Gaussian (Normal) cumulative density function:
        P = E[yₙ=1|fₙ] = Φ(fₙ)
                       = ∫ 𝓝(x|0,1) dx, where the integral is over (-∞, fₙ],
        The Normal CDF is calulcated using the error function:
                       = (1 + erf(fₙ / √2)) / 2
        for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
    """

    def __init__(self, obs_dims, link_type="logit", array_type="float32"):
        if link_type not in ["logit", "probit"]:
            raise ValueError("Link type must be logit or probit")

        super().__init__(obs_dims, obs_dims, link_type, array_type)

    def log_likelihood(self, f, y):
        """
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ ϵ ℝ
        :return:
            log p(yₙ|fₙ), p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾
        """
        # return jnp.where(jnp.equal(y, 1), self.link_fn(f), 1 - self.link_fn(f))
        lf = self.inverse_link(f)
        return jnp.where(
            jnp.equal(y, 1),
            safe_log(lf),
            safe_log(1.0 - lf),
        )

    def sample_Y(self, prng_state, f):
        """
        Bernoulli process

        :param jnp.ndarray f: input rate of shape (f_dims,)
        :returns:
            spike train of shape (obs_dims,)
        """
        p_spike = self.inverse_link(f)
        return jr.bernoulli(prng_state, p_spike).astype(self.array_dtype())


class Poisson(CountLikelihood):
    """
    Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity
    yₙ is non-negative integer count data
    """

    def __init__(self, obs_dims, tbin, link_type="log", array_type="float32"):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        if link_type not in ["log", "softplus"]:
            raise ValueError("Link type must be log or softplus")

        super().__init__(obs_dims, obs_dims, tbin, link_type, array_type)

    def log_likelihood(self, f, y):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).

        Poisson(fₙ) = μʸ exp(-μ) / yₙ!
        log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)

        :param y: observed data (yₙ) (obs_dims,)
        :param f: latent function value (fₙ) (obs_dims,)
        :return:
            log likelihood
        """
        mu = self.inverse_link(f) * self.tbin
        return distributions.Poisson_log_likelihood(mu, y)

    def variational_expectation(
        self,
        y,
        f_mean,
        f_cov,
        prng_state,
        jitter,
        approx_int_method,
        log_predictive=False,
    ):
        """
        Closed form of the expected log likelihood for exponential link function
        """
        if (
            log_predictive is False and self.link_type == LinkTypes["log"]
        ):  # closed form for E[log p(y|f)]
            f_mean = f_mean[:, 0]
            f_var = jnp.diag(f_cov)  # diagonalize
            int_exp = jnp.exp(-f_mean + f_var / 2.0)
            Ell = y * f_mean - int_exp - gammaln(y + 1.0)

            Ell = jnp.nansum(Ell)  # sum over obs_dims
            return Ell

        else:
            return super().variational_expectation(
                prng_state,
                y,
                f_mean,
                f_cov,
                prng_state,
                jitter,
                approx_int_method,
                log_predictive,
            )

    def sample_Y(self, prng_state, f):
        """
        Bernoulli process

        :param jnp.ndarray f: input rate of shape (f_dims,)
        :returns:
            spike train of shape (obs_dims,)
        """
        rate = self.inverse_link(f)
        mean = rate * self.tbin
        return distributions.sample_Poisson_counts(prng_state, mean)


class ZeroInflatedPoisson(CountLikelihood):
    """
    Zero-inflated Poisson (ZIP) count likelihood. [1]

    References:

    [1] `Untethered firing fields and intermittent silences: Why grid‐cell discharge is so variable`,
        Johannes Nagele  Andreas V.M. Herz  Martin B. Stemmler

    """

    arctanh_alpha: Union[jnp.ndarray, None]

    def __init__(self, obs_dims, tbin, alpha, link_type="log", array_type="float32"):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(obs_dims, obs_dims, tbin, link_type, array_type)
        self.arctanh_alpha = (
            jnp.arctanh(self._to_jax(alpha)) if alpha is not None else None
        )

    def apply_constraints(self):
        if self.arctanh_alpha is not None:

            def update(arctanh_alpha):
                return jnp.maximum(arctanh_alpha, 0.0)

            model = jax.tree_map(lambda p: p, self)  # copy
            model = eqx.tree_at(
                lambda tree: tree.r_inv,
                model,
                replace_fn=update,
            )
            return model

        else:
            return self

    def log_likelihood(self, f, y):
        lambd = self.inverse_link(f) * self.tbin
        alpha = jnp.tanh(self.arctanh_alpha)
        return distributions.ZIP_log_likelihood(lambd, alpha, y)

    def sample_Y(self, prng_state, f):
        """
        Sample from ZIP process.

        :param jnp.ndarray f: input rate of shape (f_dims,)
        :returns:
            spike train of shape (obs_dims,)
        """
        mean = self.inverse_link(f) * self.tbin
        alpha = jnp.tanh(self.arctanh_alpha)
        return distributions.sample_ZIP_counts(prng_state, mean, alpha)


class NegativeBinomial(CountLikelihood):
    """
    Gamma-Poisson mixture. Negative Binomial likelihood:
        p(yₙ|fₙ;rₙ) =

    Poisson case when r = infty

    :param jnp.ndarray r_inv: :math:`r^{-1}` parameter of the NB likelihood, if left to None this value is
                           expected to be provided by the heteroscedastic model in the inference class.
    """

    r_inv: Union[jnp.ndarray, None]

    def __init__(self, obs_dims, tbin, r_inv, link_type="log", array_type="float32"):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        if link_type not in ["log", "softplus"]:
            raise ValueError("Link type must be log or softplus")

        super().__init__(obs_dims, obs_dims, tbin, link_type, array_type)
        self.r_inv = self._to_jax(r_inv) if r_inv is not None else None

    def apply_constraints(self):
        if self.r_inv is not None:

            def update(r_inv):
                return jnp.maximum(r_inv, 0.0)

            model = jax.tree_map(lambda p: p, self)  # copy
            model = eqx.tree_at(
                lambda tree: tree.r_inv,
                model,
                replace_fn=update,
            )

        return model

    def log_likelihood(self, f, y):
        lambd = self.inverse_link(f) * self.tbin
        r_inv = self.r_inv  # use 1/r parameterization
        return distributions.NB_log_likelihood(lambd, r_inv, y)

    def sample_Y(self, prng_state, f):
        """
        Sample from the Gamma-Poisson mixture.

        :param jnp.ndarray f: input rate of shape (f_dims,)
        :returns:
            spike train of shape (obs_dims,)
        """
        mean = self.inverse_link(f) * self.tbin
        r = 1.0 / (self.r_inv + 1e-12)
        return distributions.sample_NB_counts(prng_state, mean, r)


class ConwayMaxwellPoisson(CountLikelihood):
    """
    Conway-Maxwell-Poisson, as described in
    https://en.wikipedia.org/wiki/Conway%E2%80%93Maxwell%E2%80%93Poisson_distribution.

    """

    log_nu: Union[jnp.ndarray, None]
    J: int

    def __init__(
        self,
        obs_dims,
        tbin,
        nu,
        J=100,
        link_type="log",
        array_type="float32",
    ):
        """
        :param int J: number of terms to use for partition function sum
        """
        super().__init__(obs_dims, obs_dims, tbin, link_type, array_type)
        self.log_nu = jnp.log(self._to_jax(nu)) if nu is not None else None
        self.J = J

    def apply_constraints(self):
        if self.log_nu is not None:

            def update(log_nu):
                return jnp.minimum(jnp.maximum(log_nu, -3.0), 3.0)  # stable range

            model = jax.tree_map(lambda p: p, self)  # copy
            model = eqx.tree_at(
                lambda tree: tree.r_inv,
                model,
                replace_fn=update,
            )
            return model

        else:
            return self

    def log_likelihood(self, f, y):
        nu = jnp.exp(self.log_nu)  # nn.functional.softplus
        lambd = self.inverse_link(f) * self.tbin
        return distributions.CMP_log_likelihood(lambd, nu, y, self.J)

    def sample_Y(self, prng_state, f):
        """
        Sample from the CMP distribution.

        :param jnp.ndarray f: input rate of shape (f_dims,)
        :returns:
            spike train of shape (obs_dims,)
        """
        mu = self.inverse_link(f) * self.tbin
        nu = jnp.exp(self.log_nu)
        return distributions.sample_CMP_counts(prng_state, mu, nu)

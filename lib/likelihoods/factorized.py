import math
from typing import Callable, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import block_diag, solve_triangular

from jax.scipy.special import erf, gammaln, logit

from ..utils.jax import mc_sample, safe_log, softplus, softplus_inv
from ..utils.linalg import gauss_hermite
from ..utils.neural import gen_CMP, gen_NB, gen_ZIP

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

        self.bias = self._to_jax(bias)  # (obs_dims,)

    def log_likelihood(self, f, y, obs_var):
        """
        Evaluate the log-Gaussian function log 𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q approximation/cubature points.

        :param jnp.ndarray y: observed data yₙ (obs_dims,)
        :param jnp.ndarray f: mean, i.e. the latent function value fₙ (obs_dims,)
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise (obs_dims,)
        """
        ll = -0.5 * (_log_twopi * safe_log(obs_var) + (y - f) ** 2 / (obs_var + 1e-10))
        return ll

    def log_likelihood(self, f, y):
        obs_var = softplus(self.pre_variance)
        return self._log_likelihood(f, y, obs_var)

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
        return f + jnp.sqrt(obs_var) * jr.normal(prng_state, shape=f_in.shape)


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
        assert link_type in ["log", "softplus"]
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

        rate = jnp.maximum(rho * self.dt, 1.0)
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
        assert link_type in ["logit", "probit"]
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
        assert link_type in ["log", "softplus"]
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
        ll = y * safe_log(mu) - mu - gammaln(y + 1)
        return ll

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
        return jr.poisson(prng_state, mean)


#             if derivatives:
#                 # dE_dm: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. mₙ of q(f)
#                 # dE_dV: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. Vₙ of q(f)
#                 dEll_dm = (y - f_mean) / obs_var
#                 dEll_dV = -0.5 / obs_var

#                 if mask is not None:
#                     dEll_dm = jnp.where(mask, 0.0, dEll_dm)  # (obs_dims, approx_points)
#                     dEll_dV = jnp.where(mask, 0.0, dEll_dV)  # (obs_dims, approx_points)

#                 dlambda_1 = (dEll_dm - 2 * dEll_dV * f_mean)[:, None]  # (f_dims, 1)
#                 dlambda_2 = jnp.diag(dEll_dV)  # (f_dims, f_dims)

#             else:
#                 dlambda_1, dlambda_2 = None, None


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

    def _log_likelihood(self, f, y, alpha):
        """
        Evaluate the Zero Inflated Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function

        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            log likelihood array (obs_dims,)
        """
        mu = self.inverse_link(f) * self.tbin

        ll_n = (
            y * safe_log(mu) - mu - gammaln(y + 1.0) + safe_log(1.0 - alpha)
        )  # -log (1-alpha)*p(N)
        p = jnp.exp(ll_n)  # stable as ll < 0
        ll_0 = safe_log(alpha + p)  # probability of zero counts, stable for alpha = 0

        ll = lax.select(y == 0, ll_0, ll_n)
        return ll

    def log_likelihood(self, f, y):
        alpha = jnp.tanh(self.arctanh_alpha)
        return self._log_likelihood(f, y, alpha)

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

    def sample_Y(self, prng_state, f):
        """
        Sample from ZIP process.

        :param jnp.ndarray f: input rate of shape (f_dims,)
        :returns:
            spike train of shape (obs_dims,)
        """
        mean = self.inverse_link(f) * self.tbin
        alpha = jnp.tanh(self.arctanh_alpha)
        return gen_ZIP(prng_state, mean, alpha)


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
        assert link_type in ["log", "softplus"]
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

    def _log_likelihood(self, f, y, r_inv):
        """
        The negative log likelihood function. Note that if disper_param is not None, it will use those values for
        the dispersion parameter rather than its own dispersion parameters.

        .. math::
            P(n|\lambda, r) = \frac{\lambda^n}{n!} \frac{\Gamma(r+n)}{\Gamma(r) \, (r+\lamba)^n} \left( 1 + \frac{\lambda}{r} \right)^{-r}

        where the mean is related to the conventional parameter :math:`\lambda = \frac{pr}{1-p}`

        For :math:`r \to \infty` we compute the likelihood as a correction to Poisson retaining :math:`\log{1 + O(r^{-1})}`.
        We parameterize the likelihood with :math:`r^{-1}`, as this allows one to reach the Poisson limit (:math:`r^{-1} \to 0`)
        using the series expansion.

        :param int b: batch index to evaluate
        :param jnp.ndarray rates: rates of shape (trial, neuron, time)
        :param jnp.ndarray n_l_rates: spikes*log(rates)
        :param jnp.ndarray spikes: spike counts of shape (trial, neuron, time)
        :param list neuron: list of neuron indices to evaluate
        :param jnp.ndarray disper_param: input for heteroscedastic NB likelihood of shape (trial, neuron, time),
                                          otherwise uses fixed :math:`r_{inv}`
        :returns:
            NLL of shape (trial, time), jnp.ndarray
        """
        asymptotic_mask = r_inv < 1e-3  # when r becomes very large
        r = 1.0 / (r_inv + asymptotic_mask)  # avoid NaNs

        mu = self.inverse_link(f) * self.tbin
        fac_gammaln = gammaln(r + y) - gammaln(r)
        fac_power = -(y + r) * safe_log(r + mu) + r * safe_log(r)

        ll_r = fac_power + fac_gammaln
        ll_r_inv = -mu - jnp.log(
            1.0 + r_inv * (y**2 + 1.0 - y * (3 / 2 + mu))
        )  # expand around small 1/r

        ll = y * safe_log(mu) - gammaln(y + 1)

        # numerically stable for large range of r
        ll += lax.select(asymptotic_mask, ll_r_inv, ll_r)
        return ll

    def log_likelihood(self, f, y):
        r_inv = self.r_inv  # use 1/r parameterization
        return self._log_likelihood(f, y, r_inv)

    def sample_Y(self, prng_state, f):
        """
        Sample from the Gamma-Poisson mixture.

        :param jnp.ndarray f: input rate of shape (f_dims,)
        :returns:
            spike train of shape (obs_dims,)
        """
        mean = self.inverse_link(f) * self.tbin
        r = 1.0 / (self.r_inv + 1e-12)
        return gen_NB(prng_state, mean, r)


class ConwayMaxwellPoisson(CountLikelihood):
    """
    Conway-Maxwell-Poisson, as described in
    https://en.wikipedia.org/wiki/Conway%E2%80%93Maxwell%E2%80%93Poisson_distribution.

    """

    log_nu: Union[jnp.ndarray, None]
    powers: jnp.ndarray
    jfact: jnp.ndarray

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

        self.powers = jnp.arange(J + 1).astype(self.array_dtype())
        self.jfact = gammaln(self.powers + 1.0)

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

    def _log_likelihood(self, f, y, nu):
        """
        :param jnp.ndarray f: input variables of shape (f_dims,)
        :returns:
            NLL of shape (obs_dims,)
        """
        log_lambda = safe_log(self.inverse_link(f) * self.tbin)

        # partition function
        log_Z_term = (
            self.powers[:, None] * log_lambda[None, :]
            - nu[None, :] * self.jfact[:, None]
        )
        log_Z = jax.nn.logsumexp(log_Z_term, axis=0)

        ll = y * log_lambda - log_Z - nu * gammaln(y + 1)
        return ll

    def log_likelihood(self, f, y):
        nu = jnp.exp(self.log_nu)  # nn.functional.softplus
        return self._log_likelihood(f, y, nu)

    def sample_Y(self, prng_state, f):
        """
        Sample from the CMP distribution.

        :param jnp.ndarray f: input rate of shape (f_dims,)
        :returns:
            spike train of shape (obs_dims,)
        """
        mu = self.inverse_link(f) * self.tbin
        nu = jnp.exp(self.log_nu)
        return gen_CMP(prng_state, mu, nu)

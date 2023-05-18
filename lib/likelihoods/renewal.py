import math
import numbers

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erf, gammainc, gammaincc, gammaln

from tensorflow_probability.substrates import jax as tfp

from tqdm.autonotebook import tqdm

from ..utils.jax import safe_log, safe_sqrt

from .base import RenewalLikelihood

_log_twopi = math.log(2 * math.pi)


class ExponentialRenewal(RenewalLikelihood):
    def __init__(
        self,
        obs_dims,
        dt,
        link_type="log",
        array_type="float32",
    ):
        """
        Renewal parameters shape can be shared for all obs_dims or independent.
        """
        super().__init__(obs_dims, dt, link_type, array_type)

    def log_density(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval array with NaN padding (obs_dims,)
        :return:
            log density of shape (obs_dims,)
        """
        ll = -ISI
        return ll

    def cum_density(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval (obs_dims,)
        """
        return 1.0 - jnp.exp(-ISI)

    def log_hazard(self, ISI):
        return jnp.zeros_like(ISI)

    def mean_scale(self):
        return 1.0

    def sample_ISI(self, prng_state):
        rd = tfp.distributions.Exponential(jnp.ones((self.obs_dims,)))
        return rd.sample(seed=prng_state)


class GammaRenewal(RenewalLikelihood):
    """
    Gamma renewal process
    """

    alpha: jnp.ndarray

    def __init__(
        self,
        obs_dims,
        dt,
        alpha,
        link_type="log",
        array_type="float32",
    ):
        """
        Renewal parameters shape can be shared for all obs_dims or independent.
        """
        super().__init__(obs_dims, dt, link_type, array_type)
        self.alpha = self._to_jax(alpha)

    def apply_constraints(self):
        """
        constrain shape parameter in numerically stable regime
        """
        model = jax.tree_map(lambda p: p, self)  # copy

        def update(alpha):
            return jnp.maximum(alpha, 1e-1)  # lower bound

        model = eqx.tree_at(
            lambda tree: tree.alpha,
            model,
            replace_fn=update,
        )

        return model

    def log_density(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval array with NaN padding (obs_dims,)
        :return:
            log density of shape (obs_dims,)
        """
        alpha = self.alpha
        ISI = ISI * alpha

        log_ISI = safe_log(ISI)
        ll = (alpha - 1) * log_ISI - ISI - gammaln(alpha)
        return ll + jnp.log(alpha)

    def cum_density(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval (obs_dims,)
        """
        alpha = self.alpha
        ISI = ISI * alpha
        return gammainc(alpha, ISI)

    def log_survival(self, ISI):
        alpha = self.alpha
        ISI = ISI * alpha

        asymptotic_mask = ISI > 2e1  # when ISI becomes very large
        asymptotic_mask = jnp.broadcast_to(asymptotic_mask, alpha.shape)

        log_ISI = safe_log(ISI)
        r_ISI = 1.0 / (ISI + 1e-12)
        log_gamma_alpha = gammaln(alpha)

        ls_stand = jnp.log(gammaincc(alpha, ISI))  # regularized incomplete Gamma

        k = jnp.arange(1, 10)[None, :]
        terms = r_ISI[:, None] ** k * jnp.exp(
            log_gamma_alpha[:, None] - gammaln(alpha[:, None] - k)
        )
        ls_asymp = (
            (alpha - 1) * log_ISI - ISI - log_gamma_alpha + jnp.log(1.0 + terms.sum(-1))
        )

        # numerically stable for large range of ISI
        return lax.select(asymptotic_mask, ls_asymp, ls_stand)

    def log_hazard(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval (obs_dims,)
        """
        alpha = self.alpha
        ISI = ISI * alpha

        asymptotic_mask = ISI > 2e1  # when ISI becomes very large
        asymptotic_mask = jnp.broadcast_to(asymptotic_mask, alpha.shape)

        log_ISI = safe_log(ISI)
        r_ISI = 1.0 / (ISI + 1e-12)
        log_gamma_alpha = gammaln(alpha)

        lh_stand = (
            (alpha - 1) * log_ISI
            - ISI
            - log_gamma_alpha
            - jnp.log(gammaincc(alpha, ISI))
        )

        k = jnp.arange(1, 10)[None, :]
        terms = r_ISI[:, None] ** k * jnp.exp(
            log_gamma_alpha[:, None] - gammaln(alpha[:, None] - k)
        )
        lh_asymp = -jnp.log(1.0 + terms.sum(-1))

        # numerically stable for large range of ISI
        return lax.select(asymptotic_mask, lh_asymp, lh_stand) + jnp.log(alpha)

    def sample_ISI(self, prng_state):
        rd = tfp.distributions.Gamma(self.alpha, self.alpha)
        return rd.sample(seed=prng_state)


class LogNormalRenewal(RenewalLikelihood):
    """
    Log-normal ISI distribution with mu = 0
    Ignores the end points of the spike train in each batch
    """

    sigma: jnp.ndarray

    def __init__(
        self,
        obs_dims,
        dt,
        sigma,
        link_type="log",
        array_type="float32",
    ):
        """
        :param np.ndarray sigma: :math:`$sigma$` parameter which is > 0
        """
        super().__init__(obs_dims, dt, link_type, array_type)
        self.sigma = self._to_jax(sigma)

    def apply_constraints(self):
        """
        constrain sigma parameter in numerically stable regime
        """
        model = jax.tree_map(lambda p: p, self)  # copy

        def update(sigma):
            return jnp.maximum(sigma, 1e-5)  # lower bound

        model = eqx.tree_at(
            lambda tree: tree.sigma,
            model,
            replace_fn=update,
        )

        return model

    def log_density(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval array with NaN padding (obs_dims,)
        :return:
            log density of shape (obs_dims,)
        """
        sigma = self.sigma
        log_scale = sigma**2 / 2.0
        ISI = ISI * jnp.exp(log_scale)

        log_ISI = safe_log(ISI)
        quad_term = -0.5 * (log_ISI / sigma) ** 2
        norm_term = -(jnp.log(sigma) + 0.5 * _log_twopi)

        ll = norm_term - log_ISI + quad_term
        return ll + log_scale

    def cum_density(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval (obs_dims,)
        """
        sigma = self.sigma
        log_scale = sigma**2 / 2.0
        log_ISI = safe_log(ISI) + log_scale  # ISI = ISI * jnp.exp(log_scale)
        return 0.5 * (1.0 + erf(log_ISI / jnp.sqrt(2.0) / self.sigma))

    def log_survival(self, ISI):
        sigma = self.sigma
        ISI = ISI * jnp.exp(sigma**2 / 2.0)
        log_ISI = safe_log(ISI)
        x = log_ISI / jnp.sqrt(2.0) / sigma

        asymptotic_mask = x > 1e1  # when ISI becomes very large
        asymptotic_mask = jnp.broadcast_to(asymptotic_mask, sigma.shape)

        ls_stand = jnp.log(1.0 - self.cum_density(ISI))

        k = jnp.arange(10)[None, :]
        terms = (-1) ** k * jnp.exp(gammaln(0.5 + k)) / x[:, None] ** (2 * k)
        ls_asymp = -(x**2) - jnp.log(x) - _log_twopi + jnp.log(terms.sum(-1))

        # numerically stable for large range of ISI
        return lax.select(asymptotic_mask, ls_asymp, ls_stand)

    def log_hazard(self, ISI):
        sigma = self.sigma
        log_scale = sigma**2 / 2.0
        ISI = ISI * jnp.exp(log_scale)

        log_ISI = safe_log(ISI)
        x = log_ISI / jnp.sqrt(2.0) / sigma
        quad_term = -(x**2)
        norm_term = -(jnp.log(sigma) + 0.5 * _log_twopi)

        ll = norm_term - log_ISI + quad_term

        asymptotic_mask = x > 1e1  # when ISI becomes very large
        asymptotic_mask = jnp.broadcast_to(asymptotic_mask, self.sigma.shape)

        lh_stand = ll - jnp.log(0.5 * (1.0 - erf(x)))

        k = jnp.arange(1, 10)[None, :]
        terms = (
            (-1) ** k
            * jnp.exp(gammaln(0.5 + k))
            / jnp.sqrt(jnp.pi)
            / x[:, None] ** (2 * k)
        )
        lh_asymp = (
            jnp.log(log_ISI) - log_ISI - 2 * jnp.log(sigma) - jnp.log(1 + terms.sum(-1))
        )

        # numerically stable for large range of ISI
        return lax.select(asymptotic_mask, lh_asymp, lh_stand) + log_scale

    def sample_ISI(self, prng_state):
        rd = tfp.distributions.LogNormal(-self.sigma**2 / 2.0, self.sigma)
        return rd.sample(seed=prng_state)


class InverseGaussianRenewal(RenewalLikelihood):
    """
    Inverse Gaussian ISI distribution with lambda = 1.
    Ignores the end points of the spike train in each batch
    """

    mu: jnp.ndarray

    def __init__(
        self,
        obs_dims,
        dt,
        mu,
        link_type="log",
        array_type="float32",
    ):
        """
        :param np.ndarray mu: :math:`$mu$` parameter which is > 0
        """
        super().__init__(obs_dims, dt, link_type, array_type)
        self.mu = self._to_jax(mu)

    def apply_constraints(self):
        """
        constrain sigma parameter in numerically stable regime
        """
        model = jax.tree_map(lambda p: p, self)  # copy

        def update(mu):
            return jnp.maximum(mu, 1e-5)  # lower bound

        model = eqx.tree_at(
            lambda tree: tree.mu,
            model,
            replace_fn=update,
        )

        return model

    def log_density(self, ISI):
        """
        :param jnp.ndarray ISI: (obs_dims,)
        """
        mu = self.mu
        ISI = ISI * mu

        log_ISI = safe_log(ISI)
        quad_term = -0.5 * (ISI / mu - 1) ** 2 / (ISI + 1e-10)
        norm_term = -0.5 * _log_twopi

        ll = norm_term - 1.5 * log_ISI + quad_term
        return ll + jnp.log(mu)

    def cum_density(self, ISI):
        mu = self.mu
        ISI = ISI * mu
        Phi = lambda x: 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))
        sqrt_ISI = safe_sqrt(ISI)
        return Phi(sqrt_ISI / mu - 1.0 / sqrt_ISI) + jnp.exp(2.0 / mu) * Phi(
            -sqrt_ISI / mu - 1.0 / sqrt_ISI
        )

    def sample_ISI(self, prng_state):
        rd = tfp.distributions.InverseGaussian(self.mu, jnp.ones_like(self.mu))
        return rd.sample(seed=prng_state) / self.mu

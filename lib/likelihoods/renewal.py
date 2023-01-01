import numbers
import math

import numpy as np
import scipy.stats as scstats

import jax
import jax.numpy as jnp
#import jax.scipy as jsc
from jax.scipy.special import erf, gammaln, gammainc

from tqdm.autonotebook import tqdm

from .base import RenewalLikelihood

_log_twopi = math.log( 2 * math.pi )





### renewal densities ###
class Gamma(RenewalLikelihood):
    """
    Gamma renewal process
    """
    alpha: jnp.ndarray

    def __init__(
        self,
        neurons,
        dt,
        alpha,
        link_type = 'log', 
        array_type = jnp.float32,
    ):
        """
        Renewal parameters shape can be shared for all neurons or independent.
        """
        super().__init__(neurons, dt, link_type, array_type)
        self.alpha = self._to_jax(alpha)

    def apply_constraints(self):
        """
        constrain shape parameter in numerically stable regime
        """
        def update(alpha):
            return jnp.minimum(jnp.maximum(alpha, 1e-5), 2.5)
        
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.alpha,
            model,
            replace_fn=update,
        )

        return model
    
    def log_renewal_density(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval array with NaN padding (obs_dims, num_isi)
        :return:
            log density of shape (obs_dims,)
        """
        num_isi = ISI.shape[-1]
        alpha = self.alpha[:, None]

        log_ISI = jnp.log(jnp.maximum(ISI, 1e-8))
        ll = (alpha - 1) * log_ISI - ISI - gammaln(alpha)
        return jnp.nansum(ll, axis=-1)
    
    def cum_renewal_density(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval (obs_dims,)
        """
        return gammainc(self.alpha, ISI)
    
    def shape_scale(self):
        return self.alpha
    
    def sample_ISI(self, num_samps):
        alpha = np.array(self.alpha)
        return scstats.gamma.rvs(alpha, scale=self.shape_scale(), size=(num_samps, self.obs_dims))

    
    
    

class LogNormal(RenewalLikelihood):
    """
    Log-normal ISI distribution with mu = 0
    Ignores the end points of the spike train in each batch
    """
    sigma: jnp.ndarray

    def __init__(
        self,
        neurons,
        dt,
        sigma,
        link_type = "log",
        array_type=jnp.float32,
    ):
        """
        :param np.ndarray sigma: :math:`$sigma$` parameter which is > 0
        """
        super().__init__(neurons, dt, link_type, array_type)
        self.sigma = self._to_jax(sigma)

    def apply_constraints(self):
        """
        constrain sigma parameter in numerically stable regime
        """
        def update(sigma):
            return jnp.maximum(sigma, 1e-5)
        
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.sigma,
            model,
            replace_fn=update,
        )

        return model
    
    def log_renewal_density(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval array with NaN padding (obs_dims, num_isi)
        :return:
            log density of shape (obs_dims,)
        """
        num_isi = ISI.shape[-1]
        sigma = self.sigma[:, None]

        log_ISI = jnp.log(jnp.maximum(ISI, 1e-8))
        quad_term = -0.5 * (log_ISI / sigma) ** 2
        norm_term = -(jnp.log(sigma) + 0.5 * _log_twopi)
        
        ll = norm_term - log_ISI + quad_term
        return jnp.nansum(ll, axis=-1)
    
    def cum_renewal_density(self, ISI):
        """
        :param jnp.ndarray ISI: interspike interval (obs_dims,)
        """
        log_ISI = jnp.log(jnp.maximum(ISI, 1e-8))
        return .5 * (1. + erf(log_ISI / jnp.sqrt(2.) / self.sigma))
    
    def shape_scale(self):
        return jnp.exp(self.sigma**2 / 2.0)

    def sample_ISI(self, num_samps):
        sigma = self.sigma
        return sctats.lognorm.rsv(s=sigma, scale=self.shape_scale(), size=(num_samps,))


    
    
class InverseGaussian(RenewalLikelihood):
    """
    Inverse Gaussian ISI distribution with lambda = 1.
    Ignores the end points of the spike train in each batch
    """
    mu: jnp.ndarray

    def __init__(
        self,
        neurons,
        dt,
        mu,
        link_type = "log",
        array_type=jnp.float32,
    ):
        """
        :param np.ndarray mu: :math:`$mu$` parameter which is > 0
        """
        super().__init__(neurons, dt, link_type, array_type)
        self.mu = self._to_jax(mu)
    
    def apply_constraints(self):
        """
        constrain sigma parameter in numerically stable regime
        """
        def update(mu):
            return jnp.maximum(mu, 1e-5)
        
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.mu,
            model,
            replace_fn=update,
        )

        return model
        
    def log_renewal_density(self, ISI):
        """
        Note the scale parameter here is the inverse of the scale parameter in nll(), as the scale
        parameter here is :math:`\tau/s` while in nll() is refers to :math:`d\tau = s*r(t) \, \mathrm{d}t`
        """
        num_isi = ISI.shape[-1]
        mu = self.mu[:, None]

        log_ISI = jnp.log(jnp.maximum(ISI, 1e-8))
        quad_term = -0.5 * (((ISI - mu) / mu) ** 2 / ISI)
        norm_term = -0.5 * _log_twopi

        ll = norm_term - 1.5 * log_ISI + quad_term
        return jnp.nansum(ll, axis=-1)
    
    def cum_renewal_density(self, ISI):
        Phi = lambda x: .5 * (1. + erf(x / jnp.sqrt(2.)))
        sqrt_ISI = jnp.sqrt(jnp.maximum(ISI, 1e-8))
        return Phi(sqrt_ISI / self.mu - 1. / sqrt_ISI) + \
            jnp.exp(2. / self.mu) * Phi(-sqrt_ISI / self.mu - 1. / sqrt_ISI)

    def shape_scale(self):
        return 1.0 / self.mu
    
    def sample_ISI(self, num_samps):
        return scstats.invgauss(mu, scale=self.shape_scale, size=(num_samps,))
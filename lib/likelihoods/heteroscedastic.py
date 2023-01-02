import math
from functools import partial

import jax
import jax.numpy as np
from jax.scipy.special import erf, gammaln

from ..utils.jax import expsum, mc_sample, safe_log, sigmoid, softplus, softplus_inv
from ..utils.linalg import gauss_hermite, get_blocks, inv

from .base import FactorizedLikelihood
from .factorized import (
    ConwayMaxwellPoisson,
    Gaussian,
    NegativeBinomial,
    ZeroInflatedPoisson,
)

_log_twopi = math.log(2 * math.pi)


class HeteroscedasticGaussian(Gaussian):
    """
    Heteroscedastic Gaussian likelihood
        p(y|f1,f2) = N(y|f1,link(f2)^2)
    """

    def __init__(self, obs_dims, array_type=jnp.float32):
        """
        :param link: link function, either 'exp' or 'softplus' (note that the link is modified with an offset)
        """
        super().__init__(obs_dims, None, array_type)
        self.f_dims = 2 * obs_dims  # overwrite

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood

        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        f_in, obs_var = f[:, 0], softplus(f[:, 1])
        return super()._log_likelihood(f_in, y, obs_var)

    def variational_expectation(
        self,
        prng_state,
        y,
        f_mean,
        f_cov,
        jitter,
        approx_int_method,
        num_approx_pts,
    ):
        # overwrite the exact homoscedastic case
        return FactorizedLikelihood.variational_expectation(
            self,
            prng_state,
            y,
            f_mean,
            f_cov,
            jitter,
            approx_int_method,
            num_approx_pts,
        )


class HeteroscedasticZeroInflatedPoisson(ZeroInflatedPoisson):
    """
    Heteroscedastic ZIP
    """

    def __init__(
        self,
        obs_dims,
        tbin,
        link_type="log",
        array_type=jnp.float32,
    ):
        super().__init__(obs_dims, tbin, None, link_type, array_type)
        self.f_dims = 2 * obs_dims  # overwrite

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood

        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        f_in, alpha = f[:, 0], jax.nn.sigmoid(f[:, 1])
        return super()._log_likelihood(f_in, y, alpha)


class HeteroscedasticNegativeBinomial(NegativeBinomial):
    """
    Heteroscedastic NB
    """

    def __init__(
        self,
        obs_dims,
        tbin,
        link_type="log",
        array_type=jnp.float32,
    ):
        super().__init__(obs_dims, tbin, None, link_type, array_type)
        self.f_dims = 2 * obs_dims  # overwrite

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood

        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        f_in, r_inv = f[:, 0], softplus(f[:, 1])
        return super()._log_likelihood(f_in, y, r_inv)


class HeteroscedasticConwayMaxwellPoisson(ConwayMaxwellPoisson):
    """
    Heteroscedastic CMP
    """

    def __init__(
        self,
        obs_dims,
        tbin,
        J=100,
        link_type="log",
        array_type=jnp.float32,
    ):
        super().__init__(obs_dims, tbin, None, J, link_type, array_type)
        self.f_dims = 2 * obs_dims  # overwrite

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood

        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        f_in, nu = f[:, 0], softplus(f[:, 1])
        return super()._log_likelihood(f_in, y, nu)


class UniversalCount(CountLikelihood):
    """
    Universal count distribution with finite cutoff at max_count
    """

    K: int
    C: int

    # final layer mapping
    W: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, obs_dims, C, K, tbin, array_type=jnp.float32):
        """
        :param int K: max spike count
        :param jnp.ndarray W: mapping matrix of shape (obs_dims, K, C)
        :param jnp.ndarray b: bias of mapping of shape (obs_dims, K)
        """
        super().__init__(obs_dims, C * obs_dims, tbin, array_type)
        self.K = K
        self.C = C

        self.W = W  # maps from NxC to NxK
        self.b = b

    def check_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for fast log-likelihood evaluation.
        Batched spikes will be a list of tensors of shape (trials, neurons, time) with trials
        set to 1 if input has no trial dimension (e.g. continuous recording).

        :param np.ndarray spikes: becomes a list of [neuron_dim, batch_dim]
        :param int/list batch_size:
        :param int filter_len: history length of the GLM couplings (1 indicates no history coupling)
        """
        if self.K < spikes.max():
            raise ValueError("Maximum count is exceeded in the spike count data")
        super().set_Y(spikes, batch_info)

    def count_logits(self, f):
        """
        Compute count probabilities from the rate model output.

        :param jnp.ndarray f: input variables of shape (f_dims,)
        """
        f_vecs = f.reshape(-1, self.C)
        a = self.W @ f_vecs + self.b  # logits (obs_dims, K+1)
        return a

    def log_likelihood(self, f, y):
        """
        Evaluate the softmax from 0 to K with linear mapping from f

        :param jnp.ndarray f: input variables of shape (f_dims,)
        :param jnp.ndarray y: count observations (obs_dims,)
        """
        a = self.count_logits(f)
        log_p_cnts = jax.nn.log_softmax(a, axis=1)
        inds = y.astype(int)
        ll = jnp.take(log_p_cnts, inds, axis=1)
        return ll

    def sample_Y(self, prng_state):
        """
        Sample from the categorical distribution.

        :param numpy.array log_probs: log count probabilities (trials, neuron, timestep, counts), no
                                      need to be normalized
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        logits = self.count_logits(f)
        return jr.categorical(prng_state, logits)


#     def _neuron_to_F(self, neuron):
#         """
#         Access subset of neurons in expanded space.
#         """
#         neuron = self._validate_neuron(neuron)
#         if len(neuron) == self.neurons:
#             F_dims = list(range(self.F_dims))
#         else: # access subset of neurons
#             F_dims = list(np.concatenate([np.arange(n*self.C, (n+1)*self.C) for n in neuron]))

#         return F_dims

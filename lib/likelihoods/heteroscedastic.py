from typing import Callable

import jax
import jax.numpy as jnp
from jax import vmap

from ..utils.jax import softplus
from . import distributions

from .base import CountLikelihood, FactorizedLikelihood


class HeteroscedasticGaussian(FactorizedLikelihood):
    """
    Heteroscedastic Gaussian likelihood
        p(y|f1,f2) = N(y|f1,link(f2)^2)
    """

    def __init__(self, obs_dims, array_type="float32"):
        """
        :param link: link function, either 'exp' or 'softplus' (note that the link is modified with an offset)
        """
        super().__init__(obs_dims, 2 * obs_dims, "none", array_type)

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood

        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        f_in, obs_var = f[:, 0], softplus(f[:, 1])
        return distributions.Gaussian_log_likelihood(f_in, obs_var, y)

    def sample_Y(self, prng_state, f):
        """
        Sample from ZIP process.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: jnp.array
        """
        f_in, obs_var = f[:, 0], softplus(f[:, 1])
        return distributions.sample_Gaussian_scalars(prng_state, f_in, obs_var)


class HeteroscedasticZeroInflatedPoisson(CountLikelihood):
    """
    Heteroscedastic ZIP
    """

    def __init__(
        self,
        obs_dims,
        tbin,
        link_type="log",
        array_type="float32",
    ):
        super().__init__(self, obs_dims, 2 * obs_dims, tbin, link_type, array_type)

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood

        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        f_in, alpha = f[:, 0], jax.nn.sigmoid(f[:, 1])
        lambd = self.inverse_link(f_in) * self.tbin
        return distributions.ZIP_log_likelihood(f_in, y, alpha)

    def sample_Y(self, prng_state, f):
        """
        Sample from ZIP process.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: jnp.array
        """
        f_in, alpha = f[:, 0], jax.nn.sigmoid(f[:, 1])
        mean = self.inverse_link(f_in) * self.tbin
        return distributions.sample_ZIP_counts(prng_state, mean, alpha)


class HeteroscedasticNegativeBinomial(CountLikelihood):
    """
    Heteroscedastic NB
    """

    def __init__(
        self,
        obs_dims,
        tbin,
        link_type="log",
        array_type="float32",
    ):
        if link_type not in ["log", "softplus"]:
            raise ValueError("Link type must be log or softplus")

        super().__init__(obs_dims, 2 * obs_dims, tbin, link_type, array_type)

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood

        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        f_in, r_inv = f[:, 0], softplus(f[:, 1])
        lambd = self.inverse_link(f_in) * self.tbin
        return distributions.NB_log_likelihood(lambd, r_inv, y)

    def sample_Y(self, prng_state, f):
        """
        Sample from ZIP process.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: jnp.array
        """
        f_in, r_inv = f[:, 0], softplus(f[:, 1])
        mean = self.inverse_link(f_in) * self.tbin
        return distributions.sample_NB_counts(prng_state, mean, r_inv)


class HeteroscedasticConwayMaxwellPoisson(CountLikelihood):
    """
    Heteroscedastic CMP
    """

    J: int

    def __init__(
        self,
        obs_dims,
        tbin,
        J=100,
        link_type="log",
        array_type="float32",
    ):
        super().__init__(self, obs_dims, 2 * obs_dims, tbin, link_type, array_type)
        self.J = J

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood

        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        f_in, nu = f[:, 0], softplus(f[:, 1])
        lambd = self.inverse_link(f_in) * self.tbin
        return distributions.CMP_log_likelihood(lambd, nu, y, self.J)

    def sample_Y(self, prng_state, f):
        """
        Sample from the CMP distribution.

        :param numpy.array rate: input rate of shape (neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: jnp.array
        """
        f_in, nu = f[:, 0], softplus(f[:, 1])
        mu = self.inverse_link(f) * self.tbin
        return distributions.sample_CMP_counts(prng_state, mu, nu)


# cannot store strings or callables in modules
BasisTypes = {
    "id": 0,
    "el": 1,
    "eq": 2,
    "ec": 3,
    "qd": 4,
}


def quad_mix(x):
    C = x.shape[-1]
    out = jnp.empty((*x.shape[:-1], C * (C - 1) // 2), dtype=x.dtype)
    k = 0
    for c in range(1, C):
        for c_ in range(c):
            out[..., k] = x[..., c] * x[..., c_]
            k += 1

    return out  # shape (..., C*(C-1)/2)


BasisFuncs = {
    0: (lambda x: x,),
    1: (lambda x: x, lambda x: jnp.exp(x)),
    2: (lambda x: x, lambda x: x**2, lambda x: jnp.exp(x)),
    3: (
        lambda x: x,
        lambda x: x**2,
        lambda x: x**3,
        lambda x: jnp.exp(x),
    ),
    4: (
        lambda x: x,
        lambda x: x**2,
        lambda x: jnp.exp(x),
        lambda x: quad_mix(x),
    ),
}


class UniversalCount(CountLikelihood):
    """
    Universal count distribution with finite cutoff at max_count
    """

    basis_expansion: int
    K: int
    C: int

    # final layer mapping
    W: jnp.ndarray
    b: jnp.ndarray

    def __init__(
        self, obs_dims, C, K, basis_expansion, W, b, tbin, array_type="float32"
    ):
        """
        :param int K: max spike count
        :param jnp.ndarray W: mapping matrix of shape (obs_dims, K+1, expand_C)
        :param jnp.ndarray b: bias of mapping of shape (obs_dims, K+1)
        """
        super().__init__(obs_dims, C * obs_dims, tbin, "none", array_type)
        self.K = K
        self.C = C
        self.basis_expansion = BasisTypes[basis_expansion]

        self.W = self._to_jax(W)  # (N, K+1, C_expand)
        self.b = self._to_jax(b)  # (N, K+1)

    def count_logits(self, f):
        """
        Compute count probabilities from the rate model output.

        :param jnp.ndarray f: input variables of shape (f_dims,)
        """
        basis_set = BasisFuncs[self.basis_expansion]

        f_vecs = f.reshape(-1, self.C)
        f_expand = jnp.concatenate([b(f_vecs) for b in basis_set], axis=-1)[..., None]
        a = (self.W @ f_expand)[..., 0] + self.b  # logits (obs_dims, K+1)
        return a

    def log_likelihood(self, f, y):
        """
        Evaluate the softmax from 0 to K with linear mapping from f

        :param jnp.ndarray f: input variables of shape (f_dims,)
        :param jnp.ndarray y: count observations (obs_dims,)
        """
        a = self.count_logits(f)
        log_p_cnts = jax.nn.log_softmax(a, axis=1)  # (obs_dims, K+1)
        inds = y.astype(jnp.dtype("int32"))  # (obs_dims)
        ll = vmap(jnp.take)(log_p_cnts, inds)
        return ll

    def sample_Y(self, prng_state, f):
        """
        Sample from the categorical distribution.

        :param numpy.array log_probs: log count probabilities (trials, neuron, timestep, counts), no
                                      need to be normalized
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        logits = self.count_logits(f)
        return jr.categorical(prng_state, logits)

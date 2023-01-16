import math
from functools import partial

import jax
import jax.numpy as jnp

from ..utils.jax import safe_log, softplus, softplus_inv
from ..utils.neural import gen_ZIP, gen_NB, gen_CMP

from .base import CountLikelihood
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
    
    def sample_Y(self, prng_state, f):
        """
        Sample from ZIP process.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: jnp.array
        """
        f_in, obs_var = f[:, 0], softplus(f[:, 1])
        return f_in + jnp.sqrt(obs_var) * jr.normal(prng_state, shape=f_in.shape)


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
    
    def sample_Y(self, prng_state, f):
        """
        Sample from ZIP process.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: jnp.array
        """
        f_in, alpha = f[:, 0], jax.nn.sigmoid(f[:, 1])
        mean = self.inverse_link(f_in) * self.tbin
        return gen_ZIP(prng_state, mean, alpha)


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
    
    def sample_Y(self, prng_state, f):
        """
        Sample from ZIP process.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: jnp.array
        """
        f_in, r_inv = f[:, 0], softplus(f[:, 1])
        mean = self.inverse_link(f_in) * self.tbin
        return gen_NB(prng_state, mean, r_inv)


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
    
    def sample_Y(self, prng_state, f):
        """
        Sample from the CMP distribution.

        :param numpy.array rate: input rate of shape (neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: jnp.array
        """
        f_in, nu = f[:, 0], softplus(f[:, 1])
        mu = self.inverse_link(f) * self.tbin
        return gen_CMP(prng_state, mu, nu)


class UniversalCount(CountLikelihood):
    """
    Universal count distribution with finite cutoff at max_count
    """

    K: int
    C: int

    # final layer mapping
    W: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, obs_dims, C, K, basis_mode, W, b, tbin, array_type=jnp.float32):
        """
        :param int K: max spike count
        :param jnp.ndarray W: mapping matrix of shape (obs_dims, K, C)
        :param jnp.ndarray b: bias of mapping of shape (obs_dims, K)
        """
        super().__init__(obs_dims, C * obs_dims, tbin, array_type)
        self.K = K
        self.C = C
        self.basis = self.get_basis(basis_mode)
        expand_C = torch.cat(
            [f_(jnp.ones(1, self.C)) for f_ in self.basis], dim=-1
        ).shape[-1]  # size of expanded vector
        
        assert W.shape == (K, expand_C)
        self.W = self._to_jax(W)  # maps from NxC_expand to NxK
        self.b = self._to_jax(b)
        
    def get_basis(basis_mode="el"):

        if basis_mode == "id":
            basis = (lambda x: x,)

        elif basis_mode == "el":  # element-wise exp-linear
            basis = (lambda x: x, lambda x: jnp.exp(x))

        elif basis_mode == "eq":  # element-wise exp-quadratic
            basis = (lambda x: x, lambda x: x**2, lambda x: jnp.exp(x))

        elif basis_mode == "ec":  # element-wise exp-cubic
            basis = (
                lambda x: x,
                lambda x: x**2,
                lambda x: x**3,
                lambda x: jnp.exp(x),
            )

        elif basis_mode == "qd":  # exp and full quadratic

            def mix(x):
                C = x.shape[-1]
                out = jnp.empty((*x.shape[:-1], C * (C - 1) // 2), dtype=x.dtype)
                k = 0
                for c in range(1, C):
                    for c_ in range(c):
                        out[..., k] = x[..., c] * x[..., c_]
                        k += 1

                return out  # shape (..., C*(C-1)/2)

            basis = (
                lambda x: x,
                lambda x: x**2,
                lambda x: jnp.exp(x),
                lambda x: mix(x),
            )

        else:
            raise ValueError("Invalid basis expansion")

        return basis

    def count_logits(self, f):
        """
        Compute count probabilities from the rate model output.

        :param jnp.ndarray f: input variables of shape (f_dims,)
        """
        f_vecs = f.reshape(-1, self.C)
        inps = F_mu.permute(0, 2, 1).reshape(samples * T, -1)  # (samplesxtime, in_dimsxchannels)
        inps = inps.view(inps.shape[0], -1, self.C)
        f_expand = torch.cat([f_(inps) for f_ in self.basis], dim=-1)
        # = self.mapping_net(inps, neuron).view(out.shape[0], -1)  # # samplesxtime, NxK
        
        a = self.W @ f_expand + self.b  # logits (obs_dims, K+1)
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

from numbers import Number

import jax
import jax.numpy as jnp

import numpy as np

from .base import Filter


class SigmoidRefractory(Filter):
    """
    Step refractory filter

    h(t) = -alpha * sigmoid(beta * (tau - t))
    """

    log_alpha: jnp.ndarray
    log_beta: jnp.ndarray
    log_tau: jnp.ndarray

    def __init__(
        self,
        alpha,
        beta,
        tau,
        filter_length,
        array_type="float32",
    ):
        """
        :param jnp.ndarray alpha: parameter alpha of shape (post, pre)
        :param jnp.ndarray beta: parameter beta of shape (post, pre)
        :param jnp.ndarray tau: timescale of shape (post, pre)
        """
        if alpha.shape != beta.shape or alpha.shape != tau.shape:
            raise ValueError("Parameters a, beta and tau must match in shape")
        if len(alpha.shape) != 2:
            raise ValueError("Parameters a must have 2 array axes (post, pre)")

        obs_dims = alpha.shape[0]
        cross_coupling = alpha.shape[1] > 1
        super().__init__(obs_dims, cross_coupling, filter_length, array_type)

        self.log_alpha = jnp.log(self._to_jax(alpha))
        self.log_beta = jnp.log(self._to_jax(beta))
        self.log_tau = jnp.log(self._to_jax(tau))

    def compute_posterior(self, mean_only, sel_outdims, jitter):
        """
        :returns:
            filter of shape (filter_length, post, pre)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.obs_dims)

        alpha = jnp.exp(self.log_alpha[sel_outdims])
        beta = jnp.exp(self.log_beta[sel_outdims])
        tau = jnp.exp(self.log_tau[sel_outdims])

        t_ = self.filter_time[::-1, None, None]
        t = beta * (tau - t_)
        h = -alpha * jax.nn.sigmoid(t)
        return h, jnp.zeros_like(h)  # (filter_length, post, pre)


class RaisedCosineBumps(Filter):
    """
    Raised cosine basis [1], takes the form of

    .. math:: f(t;a,c) = 1/2 * cos( min(max(a * log(t + c) - phi, -pi), pi) + 1 )

    References:

    [1] `Capturing the Dynamical Repertoire of Single Neurons with Generalized Linear Models`,
        Alison I. Weber, Jonathan W. Pillow (2017)

    """

    a: jnp.ndarray
    log_c: jnp.ndarray
    w: jnp.ndarray
    phi: jnp.ndarray

    def __init__(
        self,
        a,
        c,
        w,
        phi,
        filter_length,
        array_type="float32",
    ):
        """
        :param jnp.ndarray w: component weights of shape (basis, post, pre)
        :param jnp.ndarray phi: basis function parameters of shape (basis, post, pre)
        :param jnp.ndarray a: higher a is more linear spacing of peaks (post, pre)
        :param jnp.ndarray c: c shifts the peak maxima to the left (post, pre)
        """
        if w.shape != phi.shape:
            raise ValueError("Parameters w and phi must match in shape")
        if len(w.shape) != 3:
            raise ValueError("Parameters w must have 3 array axes (basis, post, pre)")

        obs_dims = w.shape[1]
        cross_coupling = w.shape[2] > 1
        super().__init__(obs_dims, cross_coupling, filter_length, array_type)
        self.a = self._to_jax(a)
        self.log_c = jnp.log(self._to_jax(c))
        self.w = self._to_jax(w)
        self.phi = self._to_jax(phi)

    def compute_posterior(self, mean_only, sel_outdims, jitter):
        """
        :return:
            filter of shape (filter_length, post, pre)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.obs_dims)

        a = self.a[sel_outdims]
        c = jnp.exp(self.log_c[sel_outdims])
        w = self.w[:, sel_outdims]
        phi = self.phi[:, sel_outdims]

        t = self.filter_time[::-1, None, None, None]
        A = jnp.minimum(
            jnp.maximum(a * jnp.log(t + c) - phi, -np.pi), np.pi
        )  # (filter_length, basis, post, pre)

        h = (w * 0.5 * (jnp.cos(A) + 1.0)).sum(1)
        return h, jnp.zeros_like(h)

from numbers import Number

import jax
import jax.numpy as jnp

import numpy as np

from ..base import ArrayTypes_

from ..GP.sparse import SparseGP

from .base import Filter


# class BayesianRaisedCosineBumps(Filter):
#     """
#     Raised cosine basis [1], takes the form of

#     .. math:: f(t;a,c) = 1/2 * cos( min(max(a * log(t + c) - phi, -pi), pi) + 1 )

#     References:

#     [1] `Capturing the Dynamical Repertoire of Single Neurons with Generalized Linear Models`,
#         Alison I. Weber, Jonathan W. Pillow (2017)

#     """

#     a: jnp.ndarray
#     log_c: jnp.ndarray
#     w: jnp.ndarray
#     phi: jnp.ndarray

#     def __init__(
#         self,
#         a,
#         c,
#         w,
#         phi,
#         filter_length,
#         array_type="float32",
#     ):
#         """
#         :param jnp.ndarray w: component weights of shape (basis, post, pre)
#         :param jnp.ndarray phi: basis function parameters of shape (basis, post, pre)
#         :param jnp.ndarray a: higher a is more linear spacing of peaks (post, pre)
#         :param jnp.ndarray c: c shifts the peak maxima to the left (post, pre)
#         """
#         if w.shape != phi.shape:
#             raise ValueError("Parameters w and phi must match in shape")
#         if len(w.shape) != 3:
#             raise ValueError("Parameters w must have 3 array axes (basis, post, pre)")

#         out_dims = w.shape[1]
#         cross_coupling = w.shape[2] > 1
#         super().__init__(out_dims, cross_coupling, filter_length, array_type)
#         self.a = self._to_jax(a)
#         self.log_c = jnp.log(self._to_jax(c))
#         self.w = self._to_jax(w)
#         self.phi = self._to_jax(phi)

#     def compute_posterior(self, mean_only, sel_outdims, jitter):
#         """
#         :return:
#             filter of shape (filter_length, post, pre)
#         """
#         if sel_outdims is None:
#             sel_outdims = jnp.arange(self.out_dims)

#         a = self.a[sel_outdims]
#         c = jnp.exp(self.log_c[sel_outdims])
#         w = self.w[:, sel_outdims]
#         phi = self.phi[:, sel_outdims]

#         t = self.filter_time[::-1, None, None, None]
#         A = jnp.minimum(
#             jnp.maximum(a * jnp.log(t + c) - phi, -np.pi), np.pi
#         )  # (filter_length, basis, post, pre)

#         h = (w * 0.5 * (jnp.cos(A) + 1.0)).sum(1)
#         return h, jnp.zeros_like(h)

#     def sample_posterior(self, prng_state, num_samps, compute_KL, sel_outdims, jitter):
#         """
#         :return:
#             filter of shape (filter_length, post, pre)
#         """
#         if sel_outdims is None:
#             sel_outdims = jnp.arange(self.out_dims)

#         t = self.filter_time[None, None, ::-1, None].repeat(num_samps, axis=0)
#         h, KL = self.gp.sample_posterior(
#             prng_state, t, compute_KL=compute_KL, sel_outdims=sel_outdims, jitter=jitter)

#         inv_tau_r = jnp.exp(-self.log_tau_r[sel_outdims])
#         if self.cross_coupling:
#             h = h.transpose(0, 2, 1).reshape(num_samps, -1, self.obs_dims, self.obs_dims)
#         else:
#             h = h.transpose(0, 2, 1)[..., None]

#         return h + self._refractory_mean(h, sel_outdims)[None], KL


class GaussianProcess(Filter):
    """
    Nonparametric GLM coupling filters. Is equivalent to multi-output GP time series. [1]

    References:

    [1] `Non-parametric generalized linear model`,
        Matthew Dowling, Yuan Zhao, Il Memming Park (2020)

    """

    out_dims: int
    gp: SparseGP
    a_r: jnp.ndarray
    log_tau_r: jnp.ndarray

    def __init__(
        self,
        gp,
        a_r,
        tau_r,
        filter_length,
    ):
        """
        :param jnp.ndarray a_r: refractory amplitude of shape (post, pre)
        """
        if a_r.shape != tau_r.shape:
            raise ValueError("Parameters a_r and tau_r must match in shape")

        cross_coupling = a_r.shape[1] > 1
        out_dims = gp.kernel.out_dims  # flattened matrix (post, pre)
        obs_dims = int(np.sqrt(out_dims)) if cross_coupling else out_dims
        super().__init__(
            obs_dims, cross_coupling, filter_length, ArrayTypes_[gp.array_type]
        )
        self.gp = gp
        self.out_dims = out_dims
        self.a_r = self._to_jax(a_r)
        self.log_tau_r = jnp.log(self._to_jax(tau_r))

    def _refractory_mean(self, h, sel_outdims):
        t = self.filter_time[::-1]
        inv_tau_r = jnp.exp(-self.log_tau_r[sel_outdims, :])
        mean = self.a_r[None, sel_outdims, :] * jnp.exp(
            -t[:, None, None] * inv_tau_r[None, ...]
        )
        return mean  # (ts, post, pre)

    def compute_posterior(self, mean_only, sel_outdims, jitter):
        """
        :return:
            filter of shape (filter_length, post, pre)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.out_dims)

        if self.cross_coupling:  # take into account (post, pre) shape
            sel_outdims_ = jnp.array(
                [(jnp.arange(self.obs_dims) + self.obs_dims * k) for k in sel_outdims]
            )
        else:
            sel_outdims_ = sel_outdims

        t = self.filter_time[None, None, ::-1, None]
        h, h_var, _, _ = self.gp.evaluate_posterior(
            t,
            mean_only=mean_only,
            diag_cov=True,
            compute_KL=False,
            compute_aux=False,
            sel_outdims=sel_outdims_,
            jitter=jitter,
        )

        if self.cross_coupling:
            h = h[0, ..., 0].T.reshape(-1, self.obs_dims, self.obs_dims)
            h_var = h_var[0, ..., 0].T.reshape(-1, self.obs_dims, self.obs_dims)
        else:
            h = h[..., 0].transpose(2, 1, 0)
            h_var = h_var[..., 0].transpose(2, 1, 0)

        return h + self._refractory_mean(h, sel_outdims), h_var

    def sample_prior(self, prng_state, num_samps, sel_outdims, jitter):
        """
        :return:
            filter of shape (num_samps, filter_length, post, pre)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.out_dims)

        if self.cross_coupling:  # take into account (post, pre) shape
            sel_outdims_ = jnp.array(
                [(jnp.arange(self.obs_dims) + self.obs_dims * k) for k in sel_outdims]
            )
        else:
            sel_outdims_ = sel_outdims

        t = self.filter_time[None, None, ::-1, None].repeat(num_samps, axis=0)
        h = self.gp.sample_prior(prng_state, t, sel_outdims=sel_outdims_, jitter=jitter)

        inv_tau_r = jnp.exp(-self.log_tau_r[sel_outdims])
        if self.cross_coupling:
            h = h.transpose(0, 2, 1).reshape(
                num_samps, -1, self.obs_dims, self.obs_dims
            )
        else:
            h = h.transpose(0, 2, 1)[..., None]

        return h + self._refractory_mean(h, sel_outdims)[None]

    def sample_posterior(self, prng_state, num_samps, compute_KL, sel_outdims, jitter):
        """
        :return:
            filter of shape (num_samps, filter_length, post, pre)
        """
        if sel_outdims is None:
            sel_outdims = jnp.arange(self.out_dims)

        if self.cross_coupling:  # take into account (post, pre) shape
            sel_outdims_ = jnp.array(
                [(jnp.arange(self.obs_dims) + self.obs_dims * k) for k in sel_outdims]
            )
        else:
            sel_outdims_ = sel_outdims

        t = self.filter_time[None, None, ::-1, None].repeat(num_samps, axis=0)
        h, KL = self.gp.sample_posterior(
            prng_state,
            t,
            compute_KL=compute_KL,
            sel_outdims=sel_outdims_,
            jitter=jitter,
        )

        inv_tau_r = jnp.exp(-self.log_tau_r[sel_outdims])
        if self.cross_coupling:
            h = h.transpose(0, 2, 1).reshape(
                num_samps, -1, self.obs_dims, self.obs_dims
            )
        else:
            h = h.transpose(0, 2, 1)[..., None]

        return h + self._refractory_mean(h, sel_outdims)[None], KL

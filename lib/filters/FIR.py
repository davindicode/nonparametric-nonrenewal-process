from numbers import Number

import jax
import jax.numpy as jnp

import numpy as np

from tqdm.autonotebook import tqdm

from ..GP.sparse import SparseGP

from .base import Filter


class SigmoidRefractory(Filter):
    """
    Step refractory filter

    h(t) = -alpha * sigmoid(beta * (tau - t))
    """

    alpha: jnp.ndarray
    beta: jnp.ndarray
    tau: jnp.ndarray

    def __init__(
        self,
        alpha,
        beta,
        tau,
        filter_length,
        array_type=jnp.float32,
    ):
        """
        :param jnp.ndarray a: parameter a of shape (post, pre)
        """
        if alpha.shape != beta.shape or alpha.shape != tau.shape:
            raise ValueError("Parameters a, beta and tau must match in shape")
        if len(alpha.shape) != 2:
            raise ValueError("Parameters a must have 2 array axes (post, pre)")

        cross_coupling = alpha.shape[1] > 1
        super().__init__(cross_coupling, filter_length, array_type)

        self.alpha = self._to_jax(alpha)
        self.beta = self._to_jax(beta)
        self.tau = self._to_jax(tau)

    def compute_filter(self, prng_state, compute_KL):
        """
        :returns:
            filter of shape (filter_length, post, pre)
        """
        t_ = self.filter_time[::-1, None, None]
        t = self.beta * (self.tau - t_)
        return -self.alpha * jax.nn.sigmoid(t), 0.0  # (filter_length, post, pre)


class RaisedCosineBumps(Filter):
    """
    Raised cosine basis [1], takes the form of

    .. math:: f(t;a,c) =

    References:

    [1] `Capturing the Dynamical Repertoire of Single Neurons with Generalized Linear Models`,
        Alison I. Weber, Jonathan W. Pillow (2017)

    """

    a: jnp.ndarray
    c: jnp.ndarray
    w: jnp.ndarray
    phi: jnp.ndarray

    def __init__(
        self,
        a,
        c,
        w,
        phi,
        filter_length,
        array_type=jnp.float32,
    ):
        """
        Raised cosine basis as used in the literature.

        .. note::

            Stimulus history object needs to assert (w_u.shape[-1] == 1 or w_u.shape[-1] == self.rate_model.out_dims)
            self.register_parameter('w_u', Parameter(torch.tensor(w_u).float())) # (basis, 1, s_len) or (basis, neurons, s_len)
            w_h # (basis, neurons, neurons)
            w_h # (basis, neurons) for self-coupling only with conv_groups = neurons
            self.register_parameter('phi_u', Parameter(torch.tensor(phi_u).float()))

        :param jnp.ndarray w: component weights of shape (basis, post, pre)
        :param jnp.ndarray phi: basis function parameters of shape (basis, post, pre)
        """
        if w.shape != phi.shape:
            raise ValueError("Parameters w and phi must match in shape")
        if len(w.shape) != 3:
            raise ValueError("Parameters w must have 3 array axes (basis, post, pre)")

        cross_coupling = w.shape[2] > 1
        super().__init__(cross_coupling, filter_length, array_type)
        self.a = self._to_jax(a)
        self.c = self._to_jax(c)
        self.w = self._to_jax(w)
        self.phi = self._to_jax(phi)

    def compute_filter(self, prng_state, compute_KL):
        """
        :returns: filter of shape (filter_length, post, pre)
        :rtype: torch.tensor
        """
        t = self.filter_time[::-1, None, None, None]
        A = jnp.minimum(
            jnp.maximum(self.a * jnp.log(t + self.c) - self.phi, -np.pi), np.pi
        )  # (filter_length, basis, post, pre)
        return (self.w * 0.5 * (jnp.cos(A) + 1.0)).sum(1), 0.0


class GaussianProcess(Filter):
    """
    Nonparametric GLM coupling filters. Is equivalent to multi-output GP time series. [1]

    References:

    [1] `Non-parametric generalized linear model`,
        Matthew Dowling, Yuan Zhao, Il Memming Park (2020)

    """

    gp: SparseGP

    def __init__(
        self,
        filter_length,
        tbin,
        array_type=jnp.float32,
    ):
        """
        :param int out_dim: the number of output dimensions per input dimension (1 or neurons)
        :param int in_dim: the number of input dimensions for the overall filter (neurons)
        """
        cross_coupling = kernel.in_dim > 1
        super().__init__(cross_coupling, filter_length, array_type)

    def compute_filter(self, prng_state, compute_KL):
        """
        :returns: filter mean of shape (n_in, n_out, filter_length), filter variance of same shape
        :rtype: tuple of torch.tensor
        """
        h_samples, KL = self.gp.sample_posterior(prng_state)
        return h_samples, KL


# class HeteroscedasticRaisedCosineBumps(Filter):
#     """
#     Raised cosine basis with weights input dependent
#     """

#     def __init__(
#         self,
#         a,
#         c,
#         phi,
#         filter_length,
#         hetero_model,
#         learnable=[True, True, True],
#         inner_loop_bs=100,
#         tensor_type=torch.float,
#     ):
#         """
#         Raised cosine basis as used in the literature, with basis function amplitudes :math:`w` given
#         by a Gaussian process.

#         :param nn.Parameter phi: basis function parameters of shape (basis, pre, post)

#         """
#         if len(phi.shape) == 2:
#             phi = phi[..., None]
#         else:
#             if len(phi.shape) != 3:
#                 raise ValueError("Parameters w must have 3 array axes")

#         conv_groups = phi.shape[1] // phi.shape[2]
#         super().__init__(filter_length, conv_groups=conv_groups, tensor_type=tensor_type)

#         if learnable[2]:
#             self.register_parameter("phi", Parameter(phi.type(self.tensor_type)))
#         else:
#             self.register_buffer("phi", phi.type(self.tensor_type))

#         if learnable[0]:
#             self.register_parameter("a", Parameter(a.type(self.tensor_type)))
#         else:
#             self.register_buffer("a", a.type(self.tensor_type))
#         if learnable[1]:
#             self.register_parameter("c", Parameter(c.type(self.tensor_type)))
#         else:
#             self.register_buffer("c", c.type(self.tensor_type))

#         self.inner_bs = inner_loop_bs
#         self.add_module("hetero_model", hetero_model)

#     def compute_filter(self, stim):
#         """
#         :param torch.Tensor stim: input covariates to the GP of shape (sample x filter_length x dims)
#         :returns: filter of shape (post, pre, filter_length)
#         :rtype: torch.tensor
#         """
#         F_mu, F_var = self.hetero_model.compute_F(stim)  # K, neurons, T

#         w = F_mu.permute(0, 2, 1).reshape(-1, *self.phi.shape)  # KxT, basis, post, pre
#         t_ = torch.arange(
#             self.filter_len, device=self.phi.device, dtype=self.tensor_type
#         )
#         t = self.filter_len - t_
#         A = torch.clamp(
#             self.a * torch.log(t + self.c)[None, None, None, :] - self.phi[..., None],
#             min=-np.pi,
#             max=np.pi,
#         )  # (basis, post, pre, filter_length)

#         filt_mean = (w[..., None] * 0.5 * (torch.cos(A[None, ...]) + 1.0)).sum(
#             1
#         )  # KxT, post, pre, filter_length
#         if isinstance(v_, Number) is False:
#             w_var = F_var.permute(0, 2, 1).reshape(-1, *self.phi.shape)
#             filt_var = (w_var[..., None] * 0.5 * (torch.cos(A[None, ...]) + 1.0)).sum(1)
#         else:
#             filt_var = 0

#         return filt_mean, filt_var

#     def KL_prior(self, importance_weighted=False):
#         return self.hetero_model.KL_prior(importance_weighted)

#     def forward(self, input, stimulus):
#         """
#         Introduces stimulus-dependent raised cosine basis. The basis function parameters are drawn from a
#         GP that depends on the stimulus values at that given time

#         :param torch.Tensor input: input spiketrain or covariates with shape (trials, neurons, filter_length)
#                                    or (samples, neurons, filter_length)
#         :param torch.Tensor stimulus: input stimulus of shape (trials, filter_length, dims)
#         :returns: filtered input of shape (trials, neurons, filter_length)
#         """
#         assert stimulus is not None
#         assert input.shape[0] == 1
#         # assert stimulus.shape[1] == input.shape[-1]-self.history_len+1

#         input_unfold = input.unfold(
#             -1, self.filter_len, 1
#         )  # samples, neurons, filter_length, fold_dim
#         stim_ = stimulus[:, self.filter_len :, :]  # samples, filter_length, dims
#         K = stim_.shape[0]
#         T = input_unfold.shape[-2]

#         inner_batches = np.ceil(T / self.inner_bs).astype(int)
#         a_ = []
#         a_var_ = []
#         for b in range(inner_batches):  # inner loop batching
#             stim_in = stim_[:, b * self.inner_bs : (b + 1) * self.inner_bs, :]
#             input_in = input_unfold[..., b * self.inner_bs : (b + 1) * self.inner_bs, :]

#             h_, v_ = self.compute_filter(stim_in)  # KxT, out, in, D_fold
#             if h_.shape[1] == 1:  # output dims
#                 a = input_in * h_[:, 0, ...].view(K, -1, *h_.shape[-2:]).permute(
#                     0, 2, 1, 3
#                 )  # K, N, T, D_fold
#                 a_.append(a.sum(-1))  # K, N, T
#             else:  # neurons
#                 a = input_in[..., None, :] * h_.view(K, -1, *h_.shape[-3:]).permute(
#                     0, 2, 1, 3, 4
#                 )  # K, N, T, n_in, D_fold
#                 a_.append(a.sum(-1).sum(-1))  # K, N, T

#         filt_var = 0 if len(a_var_) == 0 else torch.cat(a_var_, dim=-1)
#         return torch.cat(a_, dim=-1), filt_var

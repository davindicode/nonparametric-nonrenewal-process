from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from tqdm.autonotebook import tqdm

from .base import FilterLikelihood



### filters ###
class sigmoid_refractory(FilterLikelihood):
    """
    Step refractory filter
    """

    def __init__(
        self,
        a,
        tau,
        beta,
        timesteps,
        learnable=[True, True, True],
        tensor_type=torch.float,
    ):
        """ """
        if a.shape != tau.shape or a.shape != beta.shape:
            raise ValueError("Parameters a, beta and tau must match in shape")
        if len(a.shape) == 2:
            a = a[..., None]
            tau = tau[..., None]
            beta = beta[..., None]
        else:
            if len(a.shape) != 3:
                raise ValueError("Parameters a must have 3 array axes")

        conv_groups = a.shape[1] // a.shape[2]
        super().__init__(timesteps, conv_groups=conv_groups, tensor_type=tensor_type)

        if learnable[0]:
            self.register_parameter("a", Parameter(a.type(self.tensor_type)))
        else:
            self.register_buffer("a", a.type(self.tensor_type))
        if learnable[1]:
            self.register_parameter("tau", Parameter(tau.type(self.tensor_type)))
        else:
            self.register_buffer("tau", tau.type(self.tensor_type))
        if learnable[2]:
            self.register_parameter("beta", Parameter(beta.type(self.tensor_type)))
        else:
            self.register_buffer("beta", beta.type(self.tensor_type))

    def compute_filter(self):
        """
        :returns: filter of shape (post, pre, timesteps)
        :rtype: torch.tensor
        """
        t_ = torch.arange(self.filter_len, device=self.a.device, dtype=self.tensor_type)
        t = self.tau - self.filter_len + t_[None, None, :]
        return self.a * torch.sigmoid(self.beta * t)  # (post, pre, timesteps)

    def forward(self, input, stimulus=None):
        """
        Introduces the spike coupling by convolution with the spike train, no padding and left removal
        for causal convolutions.

        :param torch.Tensor input: input spiketrain or covariates with shape (trials, neurons, timesteps)
                                   or (samples, neurons, timesteps)
        :returns: filtered input of shape (trials, neurons, timesteps)
        """
        h_ = self.compute_filter()
        return F.conv1d(input, h_, groups=self.conv_groups), 0

    

class raised_cosine_bumps(FilterLikelihood):
    """
    Raised cosine basis [1], takes the form of

    .. math:: f(t;a,c) =


    References:

    [1] `Capturing the Dynamical Repertoire of Single Neurons with Generalized Linear Models`,
        Alison I. Weber, Jonathan W. Pillow (2017)

    """

    def __init__(
        self,
        a,
        c,
        phi,
        w,
        timesteps,
        learnable=[True, True, True, True],
        tensor_type=torch.float,
    ):
        """
        Raised cosine basis as used in the literature.

        .. note::

            Stimulus history object needs to assert (w_u.shape[-1] == 1 or w_u.shape[-1] == self.rate_model.out_dims)
            self.register_parameter('w_u', Parameter(torch.tensor(w_u).float())) # (basis, 1, s_len) or (basis, neurons, s_len)
            w_h # (basis, neurons, neurons)
            w_h # (basis, neurons) for self-coupling only with conv_groups = neurons
            self.register_parameter('phi_u', Parameter(torch.tensor(phi_u).float()))

            conv_groups = 1 # all-to-all couplings
            conv_groups = neurons # only self-couplings

        :param nn.Parameter w: component weights of shape (basis, post, pre)
        :param nn.Parameter phi: basis function parameters of shape (basis, post, pre)

        """
        if w.shape != phi.shape:
            raise ValueError("Parameters w and phi must match in shape")
        if len(w.shape) == 2:
            w = w[..., None]
            phi = phi[..., None]
        else:
            if len(w.shape) != 3:
                raise ValueError("Parameters w must have 3 array axes")

        conv_groups = phi.shape[1] // phi.shape[2]
        super().__init__(timesteps, conv_groups=conv_groups, tensor_type=tensor_type)

        if learnable[3]:
            self.register_parameter("w", Parameter(w.type(self.tensor_type)))
        else:
            self.register_buffer("w", w.type(self.tensor_type))
        if learnable[2]:
            self.register_parameter("phi", Parameter(phi.type(self.tensor_type)))
        else:
            self.register_buffer("phi", phi.type(self.tensor_type))

        if learnable[0]:
            self.register_parameter("a", Parameter(a.type(self.tensor_type)))
        else:
            self.register_buffer("a", a.type(self.tensor_type))
        if learnable[1]:
            self.register_parameter("c", Parameter(c.type(self.tensor_type)))
        else:
            self.register_buffer("c", c.type(self.tensor_type))

    def compute_filter(self):
        """
        :returns: filter of shape (post, pre, timesteps)
        :rtype: torch.tensor
        """
        t_ = torch.arange(self.filter_len, device=self.w.device, dtype=self.tensor_type)
        t = self.filter_len - t_
        A = torch.clamp(
            self.a * torch.log(t + self.c)[None, None, None, :] - self.phi[..., None],
            min=-np.pi,
            max=np.pi,
        )  # (basis, post, pre, timesteps)
        return (self.w[..., None] * 0.5 * (torch.cos(A) + 1.0)).sum(0)

    def forward(self, input, stimulus=None):
        """
        Introduces the spike coupling by convolution with the spike train, no padding and left removal
        for causal convolutions.

        :param torch.Tensor input: input spiketrain or covariates with shape (trials, neurons, timesteps)
                                   or (samples, neurons, timesteps)
        :returns: filtered input of shape (trials, neurons, timesteps)
        """
        h_ = self.compute_filter()
        return F.conv1d(input, h_, groups=self.conv_groups), 0


class hetero_raised_cosine_bumps(FilterLikelihood):
    """
    Raised cosine basis with weights input dependent
    """

    def __init__(
        self,
        a,
        c,
        phi,
        timesteps,
        hetero_model,
        learnable=[True, True, True],
        inner_loop_bs=100,
        tensor_type=torch.float,
    ):
        """
        Raised cosine basis as used in the literature, with basis function amplitudes :math:`w` given
        by a Gaussian process.

        :param nn.Parameter phi: basis function parameters of shape (basis, pre, post)

        """
        if len(phi.shape) == 2:
            phi = phi[..., None]
        else:
            if len(phi.shape) != 3:
                raise ValueError("Parameters w must have 3 array axes")

        conv_groups = phi.shape[1] // phi.shape[2]
        super().__init__(timesteps, conv_groups=conv_groups, tensor_type=tensor_type)

        if learnable[2]:
            self.register_parameter("phi", Parameter(phi.type(self.tensor_type)))
        else:
            self.register_buffer("phi", phi.type(self.tensor_type))

        if learnable[0]:
            self.register_parameter("a", Parameter(a.type(self.tensor_type)))
        else:
            self.register_buffer("a", a.type(self.tensor_type))
        if learnable[1]:
            self.register_parameter("c", Parameter(c.type(self.tensor_type)))
        else:
            self.register_buffer("c", c.type(self.tensor_type))

        self.inner_bs = inner_loop_bs
        self.add_module("hetero_model", hetero_model)

    def compute_filter(self, stim):
        """
        :param torch.Tensor stim: input covariates to the GP of shape (sample x timesteps x dims)
        :returns: filter of shape (post, pre, timesteps)
        :rtype: torch.tensor
        """
        F_mu, F_var = self.hetero_model.compute_F(stim)  # K, neurons, T

        w = F_mu.permute(0, 2, 1).reshape(-1, *self.phi.shape)  # KxT, basis, post, pre
        t_ = torch.arange(
            self.filter_len, device=self.phi.device, dtype=self.tensor_type
        )
        t = self.filter_len - t_
        A = torch.clamp(
            self.a * torch.log(t + self.c)[None, None, None, :] - self.phi[..., None],
            min=-np.pi,
            max=np.pi,
        )  # (basis, post, pre, timesteps)

        filt_mean = (w[..., None] * 0.5 * (torch.cos(A[None, ...]) + 1.0)).sum(
            1
        )  # KxT, post, pre, timesteps
        if isinstance(v_, Number) is False:
            w_var = F_var.permute(0, 2, 1).reshape(-1, *self.phi.shape)
            filt_var = (w_var[..., None] * 0.5 * (torch.cos(A[None, ...]) + 1.0)).sum(1)
        else:
            filt_var = 0

        return filt_mean, filt_var

    def KL_prior(self, importance_weighted=False):
        return self.hetero_model.KL_prior(importance_weighted)

    def forward(self, input, stimulus):
        """
        Introduces stimulus-dependent raised cosine basis. The basis function parameters are drawn from a
        GP that depends on the stimulus values at that given time

        :param torch.Tensor input: input spiketrain or covariates with shape (trials, neurons, timesteps)
                                   or (samples, neurons, timesteps)
        :param torch.Tensor stimulus: input stimulus of shape (trials, timesteps, dims)
        :returns: filtered input of shape (trials, neurons, timesteps)
        """
        assert stimulus is not None
        assert input.shape[0] == 1
        # assert stimulus.shape[1] == input.shape[-1]-self.history_len+1

        input_unfold = input.unfold(
            -1, self.filter_len, 1
        )  # samples, neurons, timesteps, fold_dim
        stim_ = stimulus[:, self.filter_len :, :]  # samples, timesteps, dims
        K = stim_.shape[0]
        T = input_unfold.shape[-2]

        inner_batches = np.ceil(T / self.inner_bs).astype(int)
        a_ = []
        a_var_ = []
        for b in range(inner_batches):  # inner loop batching
            stim_in = stim_[:, b * self.inner_bs : (b + 1) * self.inner_bs, :]
            input_in = input_unfold[..., b * self.inner_bs : (b + 1) * self.inner_bs, :]

            h_, v_ = self.compute_filter(stim_in)  # KxT, out, in, D_fold
            if h_.shape[1] == 1:  # output dims
                a = input_in * h_[:, 0, ...].view(K, -1, *h_.shape[-2:]).permute(
                    0, 2, 1, 3
                )  # K, N, T, D_fold
                a_.append(a.sum(-1))  # K, N, T
            else:  # neurons
                a = input_in[..., None, :] * h_.view(K, -1, *h_.shape[-3:]).permute(
                    0, 2, 1, 3, 4
                )  # K, N, T, n_in, D_fold
                a_.append(a.sum(-1).sum(-1))  # K, N, T

        filt_var = 0 if len(a_var_) == 0 else torch.cat(a_var_, dim=-1)
        return torch.cat(a_, dim=-1), filt_var


class filter_model(FilterLikelihood):
    """
    Nonparametric GLM coupling filters. Is equivalent to multi-output GP time series. [1]

    References:

    [1] `Non-parametric generalized linear model`,
        Matthew Dowling, Yuan Zhao, Il Memming Park (2020)

    """

    def __init__(
        self,
        out_dim,
        in_dim,
        timesteps,
        tbin,
        filter_model,
        num_induc=6,
        tensor_type=torch.float,
    ):
        """
        :param int out_dim: the number of output dimensions per input dimension (1 or neurons)
        :param int in_dim: the number of input dimensions for the overall filter (neurons)
        """
        conv_groups = in_dim // out_dim
        super().__init__(timesteps, conv_groups=conv_groups, tensor_type=tensor_type)
        self.out_dim = out_dim
        self.in_dim = in_dim

        if filter_model.out_dims != in_dim * out_dim:
            raise ValueError("GP output dimensions inconsistent with filter dimensions")

        self.add_module("filter_model", filter_model)
        self.register_buffer("cov", torch.arange(timesteps)[None, None, :, None] * tbin)

    def compute_filter(self):
        """
        :returns: filter mean of shape (n_in, n_out, timesteps), filter variance of same shape
        :rtype: tuple of torch.tensor
        """
        F_mu, F_var = self.filter_model.compute_F(
            self.cov
        )  # (K=1 for time series), neurons, timesteps
        if isinstance(F_var, Number) is False:
            F_var = F_var[0].view(self.in_dim, self.out_dim, -1)
        return F_mu[0].view(self.in_dim, self.out_dim, -1), F_var

    def KL_prior(self, importance_weighted=False):
        return self.filter_model.KL_prior(importance_weighted)

    def forward(self, input, stimulus=None):
        """
        Introduces the spike coupling by convolution with the spike train, no padding and left removal
        for causal convolutions.

        :param torch.Tensor input: input spiketrain or covariates with shape (trials, neurons, timesteps)
                                   or (samples, neurons, timesteps)
        :returns: filtered input of shape (trials, neurons, timesteps)
        """
        h_, v_ = self.compute_filter()
        mean_conv = F.conv1d(input, h_, groups=self.conv_groups)

        if isinstance(v_, Number) is False:
            var_conv = F.conv1d(input, v_, groups=self.conv_groups)
        else:
            var_conv = 0

        return mean_conv, var_conv


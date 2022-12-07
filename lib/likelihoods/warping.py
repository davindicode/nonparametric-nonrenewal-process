import jax.numpy as jnp


def time_transform(params, t, inverse=False):
    """
    forward transform is from tau to t
    """
    t0 = params["t0"]
    if inverse:
        t_ = -jnp.log(1 - t) * t0
        dt = t0 / (1 - t)
    else:
        s = jnp.exp(-t / t0)
        t_ = 1 - s
        dt = s / t0
    return t_, dt


def rate_rescale(self, neuron, spike_ind, rates, duplicate, minimum=1e-8):
    """
    Rate rescaling with option to dequantize, which will be random per sample.

    :param torch.Tensor rates: input rates of shape (trials, neurons, timesteps)
    :returns: list of rescaled ISIs, list index over neurons, elements of shape (trials, ISIs)
    :rtype: list
    """
    rtime = torch.cumsum(rates, dim=-1) * self.tbin
    samples = rtime.shape[0]
    rISI = []
    for tr in range(self.trials):
        isis = []
        for en, n in enumerate(neuron):
            if len(spike_ind[tr][n]) > 1:
                if self.dequant:
                    deqn = (
                        torch.rand(
                            samples, *spike_ind[tr][n].shape, device=rates.device
                        )
                        * rates[tr :: self.trials, en, spike_ind[tr][n]]
                        * self.tbin
                    )  # assume spike at 0

                    tau = rtime[tr :: self.trials, en, spike_ind[tr][n]] - deqn

                    if duplicate[tr, n]:  # re-oder in case of duplicate spike_ind
                        tau = torch.sort(tau, dim=-1)[0]

                else:
                    tau = rtime[tr :: self.trials, en, spike_ind[tr][n]]

                a = tau[:, 1:] - tau[:, :-1]
                a[a < minimum] = minimum  # don't allow near zero ISI
                isis.append(a)  # samples, order
            else:
                isis.append([])
        rISI.append(isis)

    return rISI


# class DecayingSquaredExponential(Lengthscale):
#     r"""
#     Implementation of Decaying Squared Exponential kernel:

#         :math:`k(x, z) = \exp\left(-0.5 \times \frac{|x-\beta|^2 + |z-\beta|^2} {l_{\beta}^2}\right) \,
#         \exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

#     :param torch.Tensor scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
#         kernel. Should have size 1.
#     """

#     def __init__(
#         self,
#         input_dims,
#         lengthscale,
#         lengthscale_beta,
#         beta,
#         track_dims=None,
#         f="exp",
#         tensor_type=torch.float,
#     ):
#         super().__init__(input_dims, track_dims, f, tensor_type)
#         assert (
#             self.input_dims == lengthscale.shape[0]
#         )  # lengthscale per input dimension

#         self.beta = Parameter(beta.type(tensor_type).t())  # N, D
#         self._lengthscale_beta = Parameter(
#             self.lf_inv(lengthscale_beta.type(tensor_type)).t()[:, None, None, :]
#         )  # N, K, T, D
#         self._lengthscale = Parameter(
#             self.lf_inv(lengthscale.type(tensor_type)).t()
#         )  # N, D

#     @property
#     def lengthscale(self):
#         return self.lf(self._lengthscale)[None, :, None, :]  # K, N, T, D

#     @lengthscale.setter
#     def lengthscale(self):
#         self._lengthscale.data = self.lf_inv(lengthscale)

#     @property
#     def lengthscale_beta(self):
#         return self.lf(self._lengthscale_beta)[:, None, None, :]  # N, K, T, D

#     @lengthscale_beta.setter
#     def lengthscale_beta(self):
#         return self.lf(self._lengthscale_beta)[None, :, None, :]  # K, N, T, D

#         self._lengthscale_beta.data = self.lf_inv(lengthscale_beta)

#     def forward(self, X, Z=None, diag=False):
#         X, Z = self._XZ(X, Z)

#         if diag:
#             return torch.exp(-(((X - self.beta) / self.lengthscale_beta) ** 2).sum(-1))

#         scaled_X = X / self.lengthscale  # K, N, T, D
#         scaled_Z = Z / self.lengthscale
#         X2 = (scaled_X**2).sum(-1, keepdim=True)
#         Z2 = (scaled_Z**2).sum(-1, keepdim=True)
#         XZ = scaled_X.matmul(scaled_Z.permute(0, 1, 3, 2))
#         r2 = X2 - 2 * XZ + Z2.permute(0, 1, 3, 2)

#         return torch.exp(
#             -0.5
#             * (
#                 r2
#                 + (((X - self.beta) / self.lengthscale_beta) ** 2).sum(-1)[..., None]
#                 + (((Z - self.beta) / self.lengthscale_beta) ** 2).sum(-1)[..., None, :]
#             )
#         )

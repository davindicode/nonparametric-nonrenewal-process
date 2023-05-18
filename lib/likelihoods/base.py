from typing import Union

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax import vmap
from jax.numpy.linalg import cholesky

from ..base import module
from ..utils.jax import (
    mc_sample,
    rectquad,
    rectquad_inv,
    safe_sqrt,
    softplus,
    softplus_inv,
)
from ..utils.linalg import gauss_hermite, get_blocks


# cannot store strings or callables in equinox modules
LinkTypes = {
    "none": -1,
    "log": 0,
    "softplus": 1,
    "rectified": 2,
    "rectquad": 3,
    "probit": 4,
}
LinkTypes_ = {v: k for k, v in LinkTypes.items()}

InverseLinkFuncs = {
    0: jnp.exp,
    1: softplus,
    2: lambda x: jnp.maximum(x, 0.0),
    3: rectquad,
    4: lambda x: (
        0.5 * (1.0 + jsc.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * 1e-8) + 1e-8
    ),
}

LinkFuncs = {
    0: jnp.log,
    1: softplus_inv,
    2: lambda x: x,
    3: rectquad_inv,
    4: lambda x: (
        jnp.sqrt(2.0) * jsc.special.erfinv(1.0 - 2 * (x - 1e-8) / (1 - 2 * 1e-8))
    ),
}


### classes ###
class Likelihood(module):
    """
    base class for likelihoods
    """

    f_dims: int
    obs_dims: int
    link_type: int

    def __init__(self, obs_dims, f_dims, link_type, array_type):
        """
        The logit link function:
        P = E[yₙ=1|fₙ] = 1 / 1 + exp(-fₙ)

        The Probit link function, i.e. the Error Function Likelihood:
        i.e. the Gaussian (Normal) cumulative density function:
        P = E[yₙ=1|fₙ] = Φ(fₙ)
                       = ∫ 𝓝(x|0,1) dx, where the integral is over (-∞, fₙ],
        The Normal CDF is calulcated using the error function:
                       = (1 + erf(fₙ / √2)) / 2
        for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
        """
        super().__init__(array_type)
        self.f_dims = f_dims
        self.obs_dims = obs_dims
        self.link_type = LinkTypes[link_type]

    def inverse_link(self, x):
        return InverseLinkFuncs[self.link_type](x)

    def link(self, x):
        return LinkFuncs[self.link_type](x)


class FactorizedLikelihood(Likelihood):
    """
    The likelihood model class, p(yₙ|fₙ) factorized across time points
    fₙ can be a vector (multiple parameters per observation)

    variational_expectation() computes all E_q(f) related quantities and its derivatives for NGD
    We allow multiple f (vector fₙ) to correspond to a single yₙ as in heteroscedastic likelihoods

    The default functions here use cubature/MC approximation methods, exact integration is specific
    to certain likelihood classes.
    """

    num_f_per_obs: int

    def __init__(self, obs_dims, f_dims, link_type, array_type):
        super().__init__(obs_dims, f_dims, link_type, array_type)
        self.num_f_per_obs = (
            self.f_dims // self.obs_dims
        )  # use smaller subgrid for cubature

    def log_likelihood(self, f, y):
        """
        :param jnp.ndarray f:
        """
        raise NotImplementedError(
            "direct evaluation of this log-likelihood is not implemented"
        )

    def variational_expectation(
        self,
        y,
        f_mean,
        f_cov,
        prng_state,
        jitter,
        approx_int_method,
        log_predictive=False,
        force_diagonal=False,
    ):
        """
        E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ and its derivatives
        The log marginal likelihood is log E[p(yₙ|fₙ)] = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        onlt the block-diagonal part of f_std matters

        :param jnp.array y: mean of q(f) of shape (f_dims,)
        :param jnp.array f_mean: mean of q(f) of shape (f_dims, 1)
        :param jnp.array f_cov: covariance of q(f) of shape (f_dims, f_dims)
        :return:
            weighted log likelihood at approximation locations
        """
        cubature_dim = self.num_f_per_obs  # f will have (cub_dim, approx_points)

        if approx_int_method["type"] == "GH":  # Gauss-Hermite
            f, w = gauss_hermite(cubature_dim, approx_int_method["approx_pts"])
            f = jnp.tile(
                f.T[None, ...], (self.obs_dims, 1, 1)
            )  # copy over obs_dims, (obs_dims, cubature_dim, approx_points)
        elif approx_int_method["type"] == "MC":  # sample unit Gaussian
            f, w = mc_sample(
                (self.obs_dims, cubature_dim),
                prng_state,
                approx_int_method["approx_pts"],
            )
        else:
            raise NotImplementedError("Approximate integration method not recognised")

        ### compute transformed f locations ###
        if cubature_dim == 1:
            f = f[:, 0, :]
            f_var = jnp.diag(f_cov)
            f_std = jnp.sqrt(f_var + jitter)  # safe sqrt
            df_points = f_std[:, None] * f  # (obs_dims, approx_points)

        else:  # block-diagonal form
            eps_I = jitter * jnp.eye(cubature_dim)[None, ...]

            f_cov = get_blocks(jnp.diag(jnp.diag(f_cov)), self.obs_dims, cubature_dim)

            if force_diagonal:
                chol_f_cov = safe_sqrt(f_cov)  # diagonal, more stable
            else:
                chol_f_cov = cholesky(
                    f_cov + eps_I
                )  # (obs_dims, cubature_dim, cubature_dim)

            f_mean = f_mean.reshape(self.obs_dims, 1, cubature_dim)
            df_points = (chol_f_cov @ f).transpose(
                0, 2, 1
            )  # (obs_dims, cubature_dim, approx_points)

        f_locs = (
            f_mean + df_points
        )  # integration points (obs_dims, approx_points, optional [cubature_dim])
        ll = vmap(self.log_likelihood, (1, None), 1)(
            f_locs,
            y,
        )  # vmap over approx_pts (obs_dims, approx_pts)

        if log_predictive:  # log predictive density
            log_w = jnp.log(w)
            Eq = jax.nn.logsumexp(
                jnp.nan_to_num(log_w + ll, nan=-jnp.inf), axis=0
            )  # (obs_dims,)
        else:  # expected log likelihood
            weighted_log_lik = jnp.nansum(w * ll, axis=0)  # (approx_pts,)
            Eq = jnp.sum(weighted_log_lik)  # E_q(f)[log p(y|f)]

        return Eq


#         """
#         E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ and its derivatives
#         The log marginal likelihood is log E[p(yₙ|fₙ)] = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
#         onlt the block-diagonal part of f_std matters
#         :param jnp.array f_mean: mean of q(f) of shape (f_dims,)
#         :param jnp.array f_cov: covariance of q(f) of shape (f_dims, f_dims)
#         :return:
#             log likelihood: expected log likelihood
#             dlambda_1: gradient of E_q(f)[ log p(y|f) ] w.r.t. mean natural parameter
#             dlambda_2: gradient of E_q(f)[ log p(y|f) ] w.r.t. covariance natural parameter
#         """
#         cubature_dim = self.num_f_per_out  # use smaller subgrid for cubature
#         f, w = self.approx_int_func(cubature_dim, prng_state)

#         ### compute transformed f locations ###
#         # turn f_cov into lower triangular block diagonal matrix f_
#         if cubature_dim == 1:
#             f = jnp.tile(
#                 f, (self.obs_dims, 1)
#             )  # copy over obs_dims, (obs_dims, cubature_dim)
#             f_var = jnp.diag(f_cov)
#             f_std = jnp.sqrt(f_var)
#             f_mean = f_mean[:, None]  # (f_dims, 1)
#             df_points = f_std[:, None] * f  # (obs_dims, approx_points)

#         else:  # block-diagonal form
#             f = jnp.tile(
#                 f[None, ...], (self.obs_dims, 1, 1)
#             )  # copy subgrid (obs_dims, cubature_dim, approx_points)
#             f_cov = get_blocks(np.diag(np.diag(f_cov)), self.obs_dims, cubature_dim)
#             # chol_f_cov = jnp.sqrt(np.maximum(f_cov, 1e-12)) # diagonal, more stable
#             chol_f_cov = cholesky(
#                 f_cov + jitter * jnp.eye(cubature_dim)[None, ...]
#             )  # (obs_dims, cubature_dim, cubature_dim)

#             f_mean = f_mean.reshape(self.obs_dims, cubature_dim, 1)
#             df_points = chol_f_cov @ f  # (obs_dims, cubature_dim, approx_points)

#         ### derivatives ###
#         in_shape = tree_map(lambda x: 0, lik_params)
#         if derivatives:
#             ll, dll_dm, d2ll_dm2 = vmap(
#                 self.grads_log_likelihood_n,
#                 in_axes=(0, 0, 0, in_shape, None),
#                 out_axes=(0, 0, 0),
#             )(
#                 f_mean, df_points, y, lik_params, True
#             )  # vmap over obs_dims

#             if mask is not None:  # apply mask
#                 dll_dm = jnp.where(
#                     mask[:, None], 0.0, dll_dm
#                 )  # (obs_dims, approx_points)
#                 d2ll_dm2 = jnp.where(
#                     mask[:, None], 0.0, d2ll_dm2
#                 )  # (obs_dims, approx_points)

#             dEll_dm = (w[None, :] * dll_dm).sum(1)
#             d2Ell_dm2 = (w[None, :] * d2ll_dm2).sum(1)

#             if cubature_dim == 1:  # only need diagonal f_cov
#                 dEll_dV = 0.5 * d2Ell_dm2
#                 dlambda_1 = (dEll_dm - 2 * (dEll_dV * f_mean[:, 0]))[
#                     :, None
#                 ]  # (f_dims, 1)
#                 dlambda_2 = jnp.diag(dEll_dV)  # (f_dims, f_dims)

#             else:
#                 dEll_dV = 0.5 * d2Ell_dm2[..., 0]
#                 dlambda_1 = dEll_dm[:, None] - 2 * (dEll_dV @ f_mean).reshape(
#                     -1, 1
#                 )  # (f_dims, 1)
#                 dlambda_2 = dEll_dV  # (f_dims, f_dims)

#         else:  # only compute log likelihood
#             ll, dll_dm, d2ll_dm2 = vmap(
#                 self.grads_log_likelihood_n,
#                 in_axes=(0, 0, 0, in_shape, None),
#                 out_axes=(0, 0, 0),
#             )(
#                 f_mean, df_points, y, lik_params, False
#             )  # vmap over n
#             dlambda_1, dlambda_2 = None, None

#         ### expected log likelihood ###
#         # f_mean and f_cov are from P_smoother
#         if mask is not None:  # apply mask
#             ll = jnp.where(mask[:, None], 0.0, ll)  # (obs_dims, approx_points)
#         weighted_log_lik = w * ll.sum(0)  # (approx_pts,)
#         E_log_lik = weighted_log_lik.sum()  # E_q(f)[log p(y|f)]

#         return E_log_lik, dlambda_1, dlambda_2

#     def grads_log_likelihood_n(self, f_mean, df_points, y, lik_params, derivatives):
#         """
#         Factorization over data points n, vmap over obs_dims
#         vmap over approx_points

#         :param jnp.array f_mean: (scalar) or (cubature,) for cubature_dim > 1
#         :param jnp.array df_points: (approx_points,) or (cubature, approx_points) for cubature_dim > 1

#         :return:
#             expected log likelihood ll (approx_points,)
#             dll_dm (approx_points,) or (cubature, approx_points)
#             d2ll_dm2 (approx_points,) or (cubature, cubature, approx_points)
#         """
#         f = (
#             df_points + f_mean
#         )  # (approx_points,) or (cubature, approx_points) for cubature_dim > 1

#         if derivatives:

#             def grad_func(f):
#                 ll, dll_dm = value_and_grad(self.log_likelihood_n, argnums=0)(
#                     f, y, lik_params
#                 )
#                 return dll_dm, (ll, dll_dm)

#             def temp_func(f):
#                 # dll_dm, (ll,) = grad_func(f)
#                 d2ll_dm2, aux = jacrev(grad_func, argnums=0, has_aux=True)(f)
#                 ll, dll_dm = aux
#                 return ll, dll_dm, d2ll_dm2

#             ll, dll_dm, d2ll_dm2 = vmap(temp_func, in_axes=0, out_axes=(0, 0, 0))(f)

#         else:
#             ll = vmap(self.log_likelihood_n, (0, None, None))(f, y, lik_params)
#             dll_dm, d2ll_dm2 = None, None

#         return ll, dll_dm, d2ll_dm2


class CountLikelihood(FactorizedLikelihood):
    """
    For handling count data
    """

    tbin: float

    def __init__(self, obs_dims, f_dims, tbin, link_type, array_type):
        super().__init__(obs_dims, f_dims, link_type, array_type)
        self.tbin = tbin


class RenewalLikelihood(Likelihood):
    """
    Renewal model base class
    """

    dt: float

    def __init__(
        self,
        obs_dims,
        dt,
        link_type,
        array_type,
    ):
        super().__init__(obs_dims, obs_dims, link_type, array_type)
        self.dt = dt

    def log_density(self, ISI):
        raise NotImplementedError

    def cum_density(self, ISI):
        raise NotImplementedError

    def log_survival(self, ISI):
        return jnp.log(1.0 - self.cum_density(ISI))

    def log_hazard(self, ISI):
        return self.log_density(ISI) - self.log_survival(ISI)

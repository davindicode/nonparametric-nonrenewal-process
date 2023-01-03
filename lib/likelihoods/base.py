from typing import Any, Callable, Union

import jax.numpy as jnp
import jax.scipy as jsc

from ..base import module
from ..utils.linalg import get_blocks


### classes ###
class Likelihood(module):
    """
    base class for likelihoods
    """

    f_dims: int
    obs_dims: int
    link_type: str
    inverse_link: Union[Callable, None]

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

        self.link_type = link_type
        if link_type == "log":
            self.inverse_link = lambda x: jnp.exp(x)
        elif link_type == "softplus":
            self.inverse_link = lambda x: softplus(x)
        elif link_type == "rectified":
            self.inverse_link = lambda x: jnp.maximum(x, 0.0)
        elif link_type == "logit":
            self.inverse_link = lambda x: 1 / (1 + np.exp(-x))
        elif link_type == "probit":
            jitter = 1e-8
            self.inverse_link = (
                lambda x: 0.5
                * (1.0 + jsc.special.erf(x / jnp.sqrt(2.0)))
                * (1 - 2 * jitter)
                + jitter
            )
        elif link_type == "none":
            self.inverse_link = None
        else:
            raise NotImplementedError("link function not implemented")


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
        prng_state,
        y,
        f_mean,
        f_cov,
        jitter,
        approx_int_method,
        num_approx_pts,
    ):
        """
        E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ and its derivatives
        The log marginal likelihood is log E[p(yₙ|fₙ)] = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        onlt the block-diagonal part of f_std matters

        :param np.array f_mean: mean of q(f) of shape (f_dims,)
        :param np.array f_cov: covariance of q(f) of shape (f_dims, f_dims)
        :return:
            log likelihood: expected log likelihood
        """
        cubature_dim = self.num_f_per_obs

        if approx_int_method == "GH":  # Gauss-Hermite
            f, w = gauss_hermite(cubature_dim, num_approx_pts)
        elif approx_int_method == "MC":  # sample unit Gaussian
            f, w = mc_sample(cubature_dim, prng_state, num_approx_pts)
        else:
            raise NotImplementedError("Approximate integration method not recognised")

        ### compute transformed f locations ###
        # turn f_cov into lower triangular block diagonal matrix f_
        if cubature_dim == 1:
            f = np.tile(
                f, (self.obs_dims, 1)
            )  # copy over obs_dims, (obs_dims, cubature_dim)
            f_var = np.diag(f_cov)
            f_std = np.sqrt(f_var)
            f_mean = f_mean[:, None]  # (f_dims, 1)
            df_points = f_std[:, None] * f  # (obs_dims, approx_points)

        else:  # block-diagonal form
            f = np.tile(
                f[None, ...], (self.obs_dims, 1, 1)
            )  # copy subgrid (obs_dims, cubature_dim, approx_points)
            f_cov = get_blocks(np.diag(np.diag(f_cov)), self.obs_dims, cubature_dim)
            # chol_f_cov = np.sqrt(np.maximum(f_cov, 1e-12)) # diagonal, more stable
            chol_f_cov = cholesky(
                f_cov + jitter * np.eye(cubature_dim)[None, ...]
            )  # (obs_dims, cubature_dim, cubature_dim)

            f_mean = f_mean.reshape(self.obs_dims, cubature_dim, 1)
            df_points = chol_f_cov @ f  # (obs_dims, cubature_dim, approx_points)

        f_locs = f_mean + df_points  # integration points
        ll = vmap(self.log_likelihood, (-1, None), -1)(
            f_locs, y, False
        )  # vmap over approx_pts

        # expected log likelihood
        weighted_log_lik = jnp.nansum(w * ll, axis=0)  # (approx_pts,)
        E_log_lik = jnp.nansum(weighted_log_lik)  # E_q(f)[log p(y|f)]

        return E_log_lik


#         """
#         E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ and its derivatives
#         The log marginal likelihood is log E[p(yₙ|fₙ)] = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
#         onlt the block-diagonal part of f_std matters
#         :param np.array f_mean: mean of q(f) of shape (f_dims,)
#         :param np.array f_cov: covariance of q(f) of shape (f_dims, f_dims)
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
#             f = np.tile(
#                 f, (self.obs_dims, 1)
#             )  # copy over obs_dims, (obs_dims, cubature_dim)
#             f_var = np.diag(f_cov)
#             f_std = np.sqrt(f_var)
#             f_mean = f_mean[:, None]  # (f_dims, 1)
#             df_points = f_std[:, None] * f  # (obs_dims, approx_points)

#         else:  # block-diagonal form
#             f = np.tile(
#                 f[None, ...], (self.obs_dims, 1, 1)
#             )  # copy subgrid (obs_dims, cubature_dim, approx_points)
#             f_cov = get_blocks(np.diag(np.diag(f_cov)), self.obs_dims, cubature_dim)
#             # chol_f_cov = np.sqrt(np.maximum(f_cov, 1e-12)) # diagonal, more stable
#             chol_f_cov = cholesky(
#                 f_cov + jitter * np.eye(cubature_dim)[None, ...]
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
#                 dll_dm = np.where(
#                     mask[:, None], 0.0, dll_dm
#                 )  # (obs_dims, approx_points)
#                 d2ll_dm2 = np.where(
#                     mask[:, None], 0.0, d2ll_dm2
#                 )  # (obs_dims, approx_points)

#             dEll_dm = (w[None, :] * dll_dm).sum(1)
#             d2Ell_dm2 = (w[None, :] * d2ll_dm2).sum(1)

#             if cubature_dim == 1:  # only need diagonal f_cov
#                 dEll_dV = 0.5 * d2Ell_dm2
#                 dlambda_1 = (dEll_dm - 2 * (dEll_dV * f_mean[:, 0]))[
#                     :, None
#                 ]  # (f_dims, 1)
#                 dlambda_2 = np.diag(dEll_dV)  # (f_dims, f_dims)

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
#             ll = np.where(mask[:, None], 0.0, ll)  # (obs_dims, approx_points)
#         weighted_log_lik = w * ll.sum(0)  # (approx_pts,)
#         E_log_lik = weighted_log_lik.sum()  # E_q(f)[log p(y|f)]

#         return E_log_lik, dlambda_1, dlambda_2

#     def grads_log_likelihood_n(self, f_mean, df_points, y, lik_params, derivatives):
#         """
#         Factorization over data points n, vmap over obs_dims
#         vmap over approx_points

#         :param np.array f_mean: (scalar) or (cubature,) for cubature_dim > 1
#         :param np.array df_points: (approx_points,) or (cubature, approx_points) for cubature_dim > 1

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

    def _rate_rescale(self, spikes, rates, compute_ll, return_tau):
        """
        rate rescaling, computes the log density on the way
        """
        rate_scale = self.dt / self.shape_scale()
        
        def step(carry, inputs):
            tau, ll = val
            rate, spike = inputs

            tau += rate_scale * rates[:, i]
            if compute_ll:
                ll += self.log_density(tau)
            
            return (tau.at[spike].set(0.), ll), tau if return_tau else None

        init = (jnp.zeros(self.obs_dims), jnp.zeros(self.obs_dims))
        (_, log_renewals), taus = lax.scan(step, init=init, xs=(rates, spikes))
        return log_renewals, taus
        
    def variational_expectation(
        self, y, pre_rates, 
    ):
        """
        Ignore the end points of the spike train
        
        To benefit from JIT compilation, instead of passing lists of ISIs and 
        performing rate-rescaling on them we scan temporally through the binary 
        spike train and accumulate the log density values

        :param jnp.ndarray y: binary spike train (obs_dims, ts)
        :param jnp.ndarray pre_rates: pre-link rates (obs_dims, ts)
        :param List spiketimes: list of spike time indices arrays per neuron
        :param jnp.ndarray covariates: covariates time series (mc, obs_dims, ts, in_dims)
        """
        rates = self.inverse_link(pre_rates)
        log_rates = pre_rates if self.link_type == "log" else safe_log(rates)
        spikes = (y.T > 0)
        
        log_renewals, _ = self._rate_rescale(
            spikes, rates.T, compute_ll=True, return_tau=False)
        
        ll = log_rates + log_renewals  # (obs_dims, ts)
        return ll

    def log_renewal_density(self, ISI):
        raise NotImplementedError

    def cum_renewal_density(self, ISI):
        raise NotImplementedError

    def log_survival(self, ISI):
        return jnp.log(1.0 - self.cum_renewal_density(ISI))

    def shape_scale(self):
        raise NotImplementedError

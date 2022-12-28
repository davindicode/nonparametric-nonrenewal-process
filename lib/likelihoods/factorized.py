import math
from functools import partial



import jax.numpy as np
from jax import grad, jacrev, jit, random, tree_map, value_and_grad, vmap
from jax.nn import softmax
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import block_diag, solve_triangular

from jax.scipy.special import erf, gammaln

from ..utils.jax import expsum, mc_sample, sigmoid, softplus, softplus_inv
from ..utils.linalg import gauss_hermite, get_blocks, inv

from .base import FactorizedLikelihood

_log_twopi = math.log(2 * math.pi)



### density likelihoods ###
class Gaussian(FactorizedLikelihood):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    """

    def __init__(self, out_dims, pre_variance):
        """
        :param jnp.ndarray pre_variance: The observation noise variance, σ² (out_dims,)
        """
        super().__init__(out_dims, out_dims)
        self.pre_variance = pre_variance

    @property
    def variance(self):
        return softplus(self.pre_variance)

    def log_likelihood(self, f, y):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q approximation/cubature points.

        :param y: observed data yₙ [out_dims, 1]
        :param f: mean, i.e. the latent function value fₙ [out_dims, Q]
        :param hyp: likelihood variance σ² [scalar]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise [out_dims, Q]
        """
        obs_var = jnp.maximum(softplus(self.pre_variance), 1e-8)

#         ll = jax.vmap(jax.scipy.stats.norm.logpdf, in_axes=(None, 1, None), out_axes=1)(
#             f, y, obs_var
#         )
        ll = -.5 * (_log_twopi * np.log(obs_var) + (y - f)**2 / obs_var)
        return ll

    def variational_expectation(
        self, lik_params, prng_state, jitter, y, mask, f_mean, f_cov, derivatives=True
    ):
        """
        Exact, ignore approx_int_func

        log Zₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]

        ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[log 𝓝(yₙ|fₙ,σ²)]

        :param np.array f_mean: q(f) mean with shape (f_dims,)
        :param np.array f_cov: q(f) mean with shape (f_dims, f_dims)
        """
        hyp = self.hyp if lik_params is None else lik_params
        obs_var = np.maximum(softplus(hyp["variance"]), 1e-8)
        f_var = np.diag(f_cov)  # diagonalize

        if derivatives:
            # dE_dm: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. mₙ of q(f)
            # dE_dV: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. Vₙ of q(f)
            dEll_dm = (y - f_mean) / obs_var
            dEll_dV = -0.5 / obs_var

            if mask is not None:
                dEll_dm = np.where(mask, 0.0, dEll_dm)  # (out_dims,)
                dEll_dV = np.where(mask, 0.0, dEll_dV)  # (out_dims,)

            dlambda_1 = (dEll_dm - 2 * dEll_dV * f_mean)[:, None]  # (f_dims, 1)
            dlambda_2 = np.diag(dEll_dV)  # (f_dims, f_dims)

        else:
            dlambda_1, dlambda_2 = None, None

        # ELL
        log_lik = -0.5 * (
            _log_twopi + (f_var + (y - f_mean) ** 2) / obs_var + np.log(obs_var)
        )  # (out_dims)

        # apply mask
        # mask = np.isnan(y)
        if mask is not None:
            log_lik = np.where(mask, 0.0, log_lik)  # (out_dims,)
        log_lik = log_lik.sum()  # sum over out_dims
        return log_lik, dlambda_1, dlambda_2

    

class HeteroscedasticGaussian(FactorizedLikelihood):
    """
    Heteroscedastic Gaussian likelihood
        p(y|f1,f2) = N(y|f1,link(f2)^2)
    """

    def __init__(self, out_dims, link="softplus"):
        """
        :param link: link function, either 'exp' or 'softplus' (note that the link is modified with an offset)
        """
        super().__init__(out_dims, 2 * out_dims)
        if link == "exp":
            self.link_fn = lambda x: np.exp(x)
            self.dlink_fn = lambda x: np.exp(x)
        elif link == "softplus":
            self.link_fn = lambda x: softplus(x)
            self.dlink_fn = lambda x: sigmoid(x)
        else:
            raise NotImplementedError("link function not implemented")

    @partial(jit, static_argnums=(0,))
    def log_likelihood_n(self, f, y, hyp):
        """
        Evaluate the log-likelihood
        :return:
            log likelihood of shape (approx_points,)
        """
        mu, var = f[0], np.maximum(self.link_fn(f[1]) ** 2, 1e-8)
        ll = -0.5 * ((_log_twopi + np.log(var)) + (y - mu) ** 2 / var)
        return ll


### binary likelihoods ###
class Bernoulli(FactorizedLikelihood):
    """
    Bernoulli likelihood is p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾, where P = E[yₙ=1|fₙ].
    Link function maps latent GP to [0,1].

    The probit likelihood = Bernoulli likelihood with probit link.
    The error function likelihood = probit = Bernoulli likelihood with probit link.
    The logit likelihood = Bernoulli likelihood with logit link.
    The logistic likelihood = logit = Bernoulli likelihood with logit link.

    The Probit link function, i.e. the Error Function Likelihood:
        i.e. the Gaussian (Normal) cumulative density function:
        P = E[yₙ=1|fₙ] = Φ(fₙ)
                       = ∫ 𝓝(x|0,1) dx, where the integral is over (-∞, fₙ],
        The Normal CDF is calulcated using the error function:
                       = (1 + erf(fₙ / √2)) / 2
        for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
    The logit link function:
        P = E[yₙ=1|fₙ] = 1 / 1 + exp(-fₙ)
    """

    def __init__(self, link):
        super().__init__(out_dims, out_dims, None)
        self.link = link
        if link == "logit":
            self.link_fn = lambda f: 1 / (1 + np.exp(-f))
            self.dlink_fn = lambda f: np.exp(f) / (1 + np.exp(f)) ** 2

        elif link == "probit":
            jitter = 1e-8
            self.link_fn = (
                lambda f: 0.5 * (1.0 + erf(f / np.sqrt(2.0))) * (1 - 2 * jitter)
                + jitter
            )
            self.dlink_fn = lambda f: grad(self.link_fn)(np.squeeze(f)).reshape(-1, 1)

        else:
            raise NotImplementedError("link function not implemented")

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, f, y, hyp=None):
        """
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ ϵ ℝ
        :param hyp: dummy input, Probit has no hyperparameters
        :return:
            log p(yₙ|fₙ), p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾
        """
        # return np.where(np.equal(y, 1), self.link_fn(f), 1 - self.link_fn(f))
        return np.where(
            np.equal(y, 1), np.log(self.link_fn(f)), np.log(1 - self.link_fn(f))
        )



### count likelihoods ###
class CountLikelihood(FactorizedLikelihood):
    def __init__(self, out_dims, f_dims, tbin, hyp):
        super().__init__(out_dims, f_dims, hyp)
        self.tbin = tbin
        


class Poisson(CountLikelihood):
    """
    Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity
    yₙ is non-negative integer count data
    """

    def __init__(self, out_dims, tbin, link="exp"):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(out_dims, out_dims, tbin)
        self.tbin = tbin
        self.link = link
        if link == "exp":
            self.link_fn = lambda mu: np.exp(mu)
            self.dlink_fn = lambda mu: np.exp(mu)
        elif link == "softplus":
            self.link_fn = lambda x: softplus(x)
            self.dlink_fn = lambda x: sigmoid(x)
        else:
            raise NotImplementedError("link function not implemented")

    @partial(jit, static_argnums=(0,))
    def log_likelihood_n(self, f, y, hyp=None):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :param hyp: dummy variable (Poisson has no hyperparameters)
        :return:
            Poisson(fₙ) = μʸ exp(-μ) / yₙ! [Q, 1]
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        mu = np.maximum(self.link_fn(f), 1e-8) * self.tbin
        ll = y * np.log(mu) - mu - gammaln(y + 1)
        return ll

    @partial(jit, static_argnums=(0, 8))
    def variational_expectation(
        self, lik_params, prng_state, jitter, y, mask, f_mean, f_cov, derivatives=True
    ):
        """
        Closed form of the expected log likelihood for exponential link function
        """
        if (
            False
        ):  # derivatives and self.link == 'exp':  # closed form for E[log p(y|f)]
            hyp = self.hyp if lik_params is None else lik_params
            f_var = np.diag(f_cov)  # diagonalize
            mu_mean = np.maximum(self.link_fn(f_mean), 1e-8)
            ll = y * np.log(mu_mean) - mu_mean - gammaln(y + 1)

            if derivatives:
                # dE_dm: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. mₙ of q(f)
                # dE_dV: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. Vₙ of q(f)
                dEll_dm = (y - f_mean) / obs_var
                dEll_dV = -0.5 / obs_var

                if mask is not None:
                    dEll_dm = np.where(mask, 0.0, dEll_dm)  # (out_dims, approx_points)
                    dEll_dV = np.where(mask, 0.0, dEll_dV)  # (out_dims, approx_points)

                dlambda_1 = (dEll_dm - 2 * dEll_dV * f_mean)[:, None]  # (f_dims, 1)
                dlambda_2 = np.diag(dEll_dV)  # (f_dims, f_dims)

            else:
                dlambda_1, dlambda_2 = None, None

            # ELL
            E_log_lik = -0.5 * (
                _log_twopi
                + (f_var + (y - f_mean) ** 2) / obs_var
                + np.log(np.maximum(obs_var, 1e-8))
            )  # (out_dims)

            # apply mask
            # mask = np.isnan(y)
            if mask is not None:
                log_lik = np.where(mask, 0.0, log_lik)  # (out_dims,)
            log_lik = log_lik.sum()  # sum over out_dims
            return E_log_lik, dlambda_1, dlambda_2

        else:
            return super().variational_expectation(
                lik_params, prng_state, jitter, y, mask, f_mean, f_cov, derivatives
            )


class NegativeBinomial(CountLikelihood):
    """
    NB likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    """

    def __init__(self, out_dims, tbin, link="exp"):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(out_dims, out_dims, tbin)
        self.link = link
        if link == "exp":
            self.link_fn = lambda mu: np.exp(mu)
            self.dlink_fn = lambda mu: np.exp(mu)
        elif link == "logistic":
            self.link_fn = lambda mu: softplus(mu)
            self.dlink_fn = lambda mu: sigmoid(mu)
        else:
            raise NotImplementedError("link function not implemented")

    @partial(jit, static_argnums=(0,))
    def log_likelihood_n(self, f, y, hyp=None):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :param hyp: dummy variable (Poisson has no hyperparameters)
        :return:
            Poisson(fₙ) = μʸ exp(-μ) / yₙ! [Q, 1]
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        mu = np.maximum(self.link_fn(f), 1e-8) * self.tbin
        ll = y * np.log(mu) - mu - gammaln(y + 1)
        return ll


class ZeroInflatedPoisson(CountLikelihood):
    """
    ZIP likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    """

    def __init__(self, out_dims, tbin, link="exp"):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(out_dims, out_dims, tbin, hyp)
        self.link = link
        if link == "exp":
            self.link_fn = lambda mu: np.exp(mu)
            self.dlink_fn = lambda mu: np.exp(mu)
        elif link == "logistic":
            self.link_fn = lambda mu: softplus(mu)
            self.dlink_fn = lambda mu: sigmoid(mu)
        else:
            raise NotImplementedError("link function not implemented")

    @partial(jit, static_argnums=(0,))
    def log_likelihood_n(self, f, y, hyp=None):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :param hyp: dummy variable (Poisson has no hyperparameters)
        :return:
            Poisson(fₙ) = μʸ exp(-μ) / yₙ! [Q, 1]
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        mu = np.maximum(self.link_fn(f), 1e-8) * self.tbin
        ll = y * np.log(mu) - mu - gammaln(y + 1)
        return ll


class UniversalCount(CountLikelihood):
    """
    Universal count likelihood
    """

    def __init__(self, out_dims, C, K, tbin):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(out_dims, C * out_dims, tbin)
        self.K = K

    @partial(jit, static_argnums=(0,))
    def log_likelihood_n(self, f, y, hyp=None):
        """
        Evaluate the softmax from 0 to K with linear mapping from f
        """
        hyp = self.hyp if params is None else params
        W = hyp["W"]
        b = hyp["b"]
        logp_cnts = log_softmax(W @ f + b)
        return logp_cnts[int(y)]

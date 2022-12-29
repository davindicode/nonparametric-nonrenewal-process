from typing import Callable
import math

import jax.numpy as jnp
from jax import random, vmap
from jax.nn import softmax
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import block_diag, solve_triangular

from jax.scipy.special import erf, gammaln

from ..utils.jax import expsum, mc_sample, sigmoid, softplus, softplus_inv
from ..utils.linalg import gauss_hermite, get_blocks, inv_PSD

from .base import FactorizedLikelihood, CountLikelihood

_log_twopi = math.log(2 * math.pi)



### density likelihoods ###
class Gaussian(FactorizedLikelihood):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    """
    pre_variance: jnp.ndarray

    def __init__(self, out_dims, pre_variance, array_type=jnp.float32):
        """
        :param jnp.ndarray pre_variance: The observation noise variance, σ² (out_dims,)
        """
        super().__init__(out_dims, out_dims, array_type)
        self.pre_variance = pre_variance

    @property
    def variance(self):
        return softplus(self.pre_variance)

    def log_likelihood(self, f, y):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q approximation/cubature points.

        :param jnp.ndarray y: observed data yₙ (out_dims, 1)
        :param jnp.ndarray f: mean, i.e. the latent function value fₙ (out_dims, Q)
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise (out_dims, Q)
        """
        obs_var = jnp.maximum(softplus(self.pre_variance), 1e-8)[:, None]

#         ll = jax.vmap(jax.scipy.stats.norm.logpdf, in_axes=(None, 1, None), out_axes=1)(
#             f, y, obs_var
#         )
        ll = -.5 * (_log_twopi * jnp.log(obs_var) + (y - f)**2 / obs_var)
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
        obs_var = np.maximum(softplus(self.pre_variance), 1e-8)
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
    
    

class LogCoxProcess(FactorizedLikelihood):
    """
    The continuous-time point process with log intensity given by stochastic process
    
    Small time bin discrete approximation
    """
    link_fn: Callable
    dt: float
    
    def __init__(
        self, neurons, dt, link_fn, array_type=jnp.float32, 
    ):
        super().__init__(neurons, neurons, array_type)
        self.dt = dt
        self.link_fn = link_fn

    def sample_helper(self, h, b, neuron, samples):
        """
        NLL helper function for sample evaluation. Note that spikes is batched including history
        when the model uses history couplings, hence we sample the spike batches without the
        history segments from this function.
        """
        rates = self.f(h)  # watch out for underflow or overflow here
        batch_edge, _, _ = self.batch_info
        spikes = self.all_spikes[:, neuron, batch_edge[b] : batch_edge[b + 1]].to(
            self.tbin.device
        )
        if (
            self.trials != 1 and samples > 1 and self.trials < h.shape[0]
        ):  # cannot rely on broadcasting
            spikes = spikes.repeat(samples, 1, 1)  # MC x trials

        n_l_rates = spikes * h  # spike count times log rate

        return rates, n_l_rates, spikes

    def nll(self, b, rates, n_l_rates):
        nll = -n_l_rates + rates
        return nll.sum(1)

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        """
        Computes the terms for variational expectation :math:`\mathbb{E}_{q(f)q(z)}[]`, which
        can be used to compute different likelihood objectives.
        The returned tensor will have sample dimension as MC over :math:`q(z)`, depending
        on the evaluation mode will be MC or GH or exact over the likelihood samples. This
        is all combined in to the same dimension to be summed over. The weights :math:`w_s`
        are the quadrature weights or equal weights for MC, with appropriate normalization.

        :param int samples: number of MC samples or GH points (exact will ignore and give 1)

        :returns: negative likelihood term of shape (samples, timesteps), sample weights (samples, 1
        :rtype: tuple of jnp.tensors
        """
        if mode == "MC":
            h = self.mc_gen(F_mu, F_var, samples, neuron)  # h has only observed neurons
            rates, n_l_rates, spikes = self.sample_helper(h, b, neuron, samples)
            ws = jnp.tensor(1.0 / rates.shape[0])
            
        elif mode == "GH":
            h, ws = self.gh_gen(F_mu, F_var, samples, neuron)
            rates, n_l_rates, spikes = self.sample_helper(h, b, neuron, samples)
            ws = ws[:, None]
            
        elif mode == "direct":
            rates, n_l_rates, spikes = self.sample_helper(
                F_mu[:, neuron, :], b, neuron, samples
            )
            ws = jnp.tensor(1.0 / rates.shape[0])
            
        else:
            raise NotImplementedError

        return self.nll(b, rates, n_l_rates), ws




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
    
    link_fn: Callable

    def __init__(self, out_dims, link_fn, array_type=jnp.float32):
        super().__init__(out_dims, out_dims, array_type)
        
        if link_fn == "logit":
            self.link_fn = lambda f: 1 / (1 + np.exp(-f))
            #self.dlink_fn = lambda f: np.exp(f) / (1 + np.exp(f)) ** 2

        elif link_fn == "probit":
            jitter = 1e-8
            self.link_fn = (
                lambda f: 0.5 * (1.0 + erf(f / np.sqrt(2.0))) * (1 - 2 * jitter)
                + jitter
            )
            #self.dlink_fn = lambda f: grad(self.link_fn)(np.squeeze(f)).reshape(-1, 1)

        else:
            raise NotImplementedError("link function not implemented")

    def log_likelihood(self, f, y):
        """
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ ϵ ℝ
        :return:
            log p(yₙ|fₙ), p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾
        """
        # return np.where(np.equal(y, 1), self.link_fn(f), 1 - self.link_fn(f))
        return np.where(
            np.equal(y, 1), np.log(self.link_fn(f)), np.log(1 - self.link_fn(f))
        )
    
    def sample(self, rate, neuron=None, XZ=None):
        """
        Approximate by a Bernoulli process, slight bias introduced as spike means at least one spike

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        return jr.bernoulli(rate)

        


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

    def log_likelihood_n(self, f, y):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            Poisson(fₙ) = μʸ exp(-μ) / yₙ! [Q, 1]
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        mu = np.maximum(self.link_fn(f), 1e-8) * self.tbin
        ll = y * np.log(mu) - mu - gammaln(y + 1)
        return ll

    def variational_expectation(
        self, lik_params, prng_state, jitter, y, mask, f_mean, f_cov, derivatives=True
    ):
        """
        Closed form of the expected log likelihood for exponential link function
        """
        if (
            False
        ):  # derivatives and self.link == 'exp':  # closed form for E[log p(y|f)]
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

    def log_likelihood_n(self, f, y):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
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

    def log_likelihood_n(self, f, y):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
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

    def log_likelihood_n(self, f, y):
        """
        Evaluate the softmax from 0 to K with linear mapping from f
        """
        logp_cnts = log_softmax(W @ f + b)
        return logp_cnts[int(y)]
    
    
    
def gen_CMP(mu, nu, max_rejections=1000):
    """
    Use rejection sampling to sample from the COM-Poisson count distribution. [1]

    References:

    [1] `Bayesian Inference, Model Selection and Likelihood Estimation using Fast Rejection
         Sampling: The Conway-Maxwell-Poisson Distribution`, Alan Benson, Nial Friel (2021)

    :param numpy.array rate: input rate of shape (..., time)
    :param float tbin: time bin size
    :param float eps: order of magnitude of P(N>1)/P(N<2) per dilated Bernoulli bin
    :param int max_count: maximum number of spike counts per bin possible
    :returns: inhomogeneous Poisson process sample
    :rtype: numpy.array
    """
    trials = mu.shape[0]
    neurons = mu.shape[1]
    Y = np.empty(mu.shape)

    for tr in range(trials):
        for n in range(neurons):
            mu_, nu_ = mu[tr, n, :], nu[tr, n, :]

            # Poisson
            k = 0
            left_bins = np.where(nu_ >= 1)[0]
            while len(left_bins) > 0:
                mu__, nu__ = mu_[left_bins], nu_[left_bins]
                y_dash = jnp.poisson(jnp.tensor(mu__)).numpy()
                _mu_ = np.floor(mu__)
                alpha = (
                    mu__ ** (y_dash - _mu_)
                    / scsps.factorial(y_dash)
                    * scsps.factorial(_mu_)
                ) ** (nu__ - 1)

                u = np.random.rand(*mu__.shape)
                selected = u <= alpha
                Y[tr, n, left_bins[selected]] = y_dash[selected]
                left_bins = left_bins[~selected]
                if k >= max_rejections:
                    raise ValueError("Maximum rejection steps exceeded")
                else:
                    k += 1

            # geometric
            k = 0
            left_bins = np.where(nu_ < 1)[0]
            while len(left_bins) > 0:
                mu__, nu__ = mu_[left_bins], nu_[left_bins]
                p = 2 * nu__ / (2 * mu__ * nu__ + 1 + nu__)
                u_0 = np.random.rand(*p.shape)

                y_dash = np.floor(np.log(u_0) / np.log(1 - p))
                a = np.floor(mu__ / (1 - p) ** (1 / nu__))
                alpha = (1 - p) ** (a - y_dash) * (
                    mu__ ** (y_dash - a) / scsps.factorial(y_dash) * scsps.factorial(a)
                ) ** nu__

                u = np.random.rand(*mu__.shape)
                selected = u <= alpha
                Y[tr, n, left_bins[selected]] = y_dash[selected]
                left_bins = left_bins[~selected]
                if k >= max_rejections:
                    raise ValueError("Maximum rejection steps exceeded")
                else:
                    k += 1

    return Y



class ConwayMaxwellPoisson(CountLikelihood):
    """
    Conway-Maxwell-Poisson, as described in
    https://en.wikipedia.org/wiki/Conway%E2%80%93Maxwell%E2%80%93Poisson_distribution.

    """
    log_nu: jnp.ndarray
    powers: jnp.ndarray
    jfact: jnp.ndarray

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        log_nu,
        array_type=jnp.float32,
        J=100,
        strict_likelihood=True,
    ):
        super().__init__(tbin, neurons, inv_link, array_type, strict_likelihood)
        self.log_nu = jnp.array(log_nu, dtype=self.array_type)

        self.powers = jnp.arange(J + 1).astype(self.array_type)
        self.jfact = gammaln(self.powers + 1.0)

    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        if self.strict_likelihood:
            tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
            if (
                tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]
            ):  # cannot rely on broadcasting
                tfact = tfact.repeat(spikes.shape[0] // tfact.shape[0], 1, 1)
        else:
            tfact = 0

        lfact = self.lfact[b][:, neuron, :].to(self.tbin.device)
        if (
            lfact.shape[0] != 1 and lfact.shape[0] < spikes.shape[0]
        ):  # cannot rely on broadcasting
            lfact = lfact.repeat(spikes.shape[0] // lfact.shape[0], 1, 1)

        return tfact, lfact

    def log_Z(self, log_lambda, nu):
        """
        Partition function.

        :param jnp.Tensor lambd: lambda of shape (samples, neurons, timesteps)
        :param jnp.Tensor nu: nu of shape (samples, neurons, timesteps)
        :returns: log Z of shape (samples, neurons, timesteps)
        :rtype: jnp.tensor
        """
        # indx = jnp.where((self.powers*lambd.max() - nu_.min()*self.j) < -1e1) # adaptive
        # if len(indx) == 0:
        #    indx = self.J+1
        log_Z_term = (
            self.powers[:, None, None, None] * log_lambda[None, ...]
            - nu[None, ...] * self.jfact[:, None, None, None]
        )
        return jax.nn.logsumexp(log_Z_term, axis=0)

    def nll(self, b, rates, n_l_rates, spikes, neuron, disper_param=None):
        """
        :param int b: batch index to evaluate
        :param jnp.Tensor rates: rates of shape (trial, neuron, time)
        :param jnp.Tensor n_l_rates: spikes*log(rates)
        :param jnp.Tensor spikes: spike counts of shape (neuron, time)
        :param list neuron: list of neuron indices to evaluate
        :param jnp.Tensor disper_param: input for heteroscedastic NB likelihood of shape (trial, neuron, time),
                                          otherwise uses fixed :math:`\nu`
        :returns: NLL of shape (trial, time)
        :rtype: jnp.tensor
        """
        if disper_param is None:
            nu_ = jnp.exp(self.log_nu).expand(1, self.neurons)[:, neuron, None]
        else:
            nu_ = jnp.exp(disper_param)  # nn.functional.softplus

        tfact, lfact = self.get_saved_factors(b, neuron, spikes)
        log_lambda = jnp.log(rates * self.tbin + 1e-12)

        l_Z = self.log_Z(log_lambda, nu_)

        nll = -n_l_rates + l_Z - tfact + nu_ * lfact
        return nll.sum(1)

    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample from the CMP distribution.

        :param numpy.array rate: input rate of shape (neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        mu_ = rate[:, neuron, :] * self.tbin.item()

        if self.dispersion_mapping is None:
            nu_ = (
                jnp.exp(self.log_nu)[None, :, None]
                .expand(rate.shape[0], self.neurons, mu_.shape[-1])
                .data.cpu()
                .numpy()[:, neuron, :]
            )
        else:
            samples = rate.shape[0]
            with jnp.no_grad():
                disp = self.sample_dispersion(XZ, rate.shape[0] // XZ.shape[0], neuron)
            nu_ = jnp.exp(disp).cpu().numpy()

        return gen_CMP(mu_, nu_)
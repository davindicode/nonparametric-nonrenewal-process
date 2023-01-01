from typing import Callable, Union
import math

import jax.numpy as jnp
from jax import random, vmap
from jax.nn import softmax, sigmoid
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import block_diag, solve_triangular

from jax.scipy.special import erf, gammaln, logit

from ..utils.jax import expsum, mc_sample, softplus, softplus_inv
from ..utils.linalg import gauss_hermite, get_blocks, inv_PSD
from ..utils.neural import gen_CMP

from .base import FactorizedLikelihood, CountLikelihood

_log_twopi = math.log(2 * math.pi)



### density likelihoods ###
class Gaussian(FactorizedLikelihood):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    """
    pre_variance: Union[jnp.ndarray, None]

    def __init__(self, out_dims, pre_variance, array_type=jnp.float32):
        """
        :param jnp.ndarray pre_variance: The observation noise variance, σ² (out_dims,)
        """
        super().__init__(out_dims, out_dims, 'none', array_type)
        self.pre_variance = pre_variance

    @property
    def variance(self):
        return softplus(self.pre_variance)

    def log_likelihood(self, f, y):
        """
        Evaluate the log-Gaussian function log 𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q approximation/cubature points.

        :param jnp.ndarray y: observed data yₙ (out_dims,)
        :param jnp.ndarray f: mean, i.e. the latent function value fₙ (out_dims,)
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise (out_dims,)
        """
        obs_var = jnp.maximum(softplus(self.pre_variance), 1e-8)
#         ll = jax.vmap(jax.scipy.stats.norm.logpdf, in_axes=(None, 1, None), out_axes=1)(
#             f, y, obs_var
#         )
        ll = -.5 * (_log_twopi * jnp.log(obs_var) + (y - f)**2 / obs_var)
        return ll

    def variational_expectation(
        self, prng_state, y, f_mean, f_cov, jitter, approx_int_method, num_approx_pts, 
    ):
        """
        Exact integration, overwrite approximate integration

        log Zₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]

        ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[log 𝓝(yₙ|fₙ,σ²)]

        :param np.array f_mean: q(f) mean with shape (f_dims,)
        :param np.array f_cov: q(f) mean with shape (f_dims, f_dims)
        """
        obs_var = np.maximum(softplus(self.pre_variance), 1e-8)
        f_var = np.diag(f_cov)  # diagonalize

        # ELL
        log_lik = -0.5 * (
            _log_twopi + (f_var + (y - f_mean) ** 2) / obs_var + np.log(obs_var)
        )  # (out_dims)
        
        #log_lik = np.where(mask, 0.0, log_lik)  # (out_dims,)
        log_lik = log_lik.sum()  # sum over out_dims
        return log_lik, dlambda_1, dlambda_2
    
    def sample_Y(self, prng_state, rate):
        """
        Gaussian

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        return rate * jr.normal(prng_state)
    
    
#     """
#     Exact, ignore approx_int_func

#     log Zₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]

#     ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[log 𝓝(yₙ|fₙ,σ²)]

#     :param np.array f_mean: q(f) mean with shape (f_dims,)
#     :param np.array f_cov: q(f) mean with shape (f_dims, f_dims)
#     """
#         if derivatives:
#             # dE_dm: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. mₙ of q(f)
#             # dE_dV: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. Vₙ of q(f)
#             dEll_dm = (y - f_mean) / obs_var
#             dEll_dV = -0.5 / obs_var

#             if mask is not None:
#                 dEll_dm = np.where(mask, 0.0, dEll_dm)  # (out_dims,)
#                 dEll_dV = np.where(mask, 0.0, dEll_dV)  # (out_dims,)

#             dlambda_1 = (dEll_dm - 2 * dEll_dV * f_mean)[:, None]  # (f_dims, 1)
#             dlambda_2 = np.diag(dEll_dV)  # (f_dims, f_dims)

#         else:
#             dlambda_1, dlambda_2 = None, None

    
    
    

class LogCoxProcess(FactorizedLikelihood):
    """
    The continuous-time point process with log intensity given by stochastic process
    
    Small time bin discrete approximation
    """
    dt: float
    
    def __init__(
        self, neurons, dt, array_type=jnp.float32, 
    ):
        super().__init__(neurons, neurons, 'none', array_type)
        self.dt = dt
        
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
    
    def sample_Y(self, rate):
        """
        Bernoulli process approximation with small dt

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        return jr.bernoulli(prng_state, rate)



### discrete likelihoods ###
class Bernoulli(FactorizedLikelihood):
    """
    Bernoulli likelihood is p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾, where P = E[yₙ=1|fₙ].
    Link function maps latent GP to [0,1].
    
    The error function likelihood = probit = Bernoulli likelihood with probit link.
    The logistic likelihood = logit = Bernoulli likelihood with logit link.
    """

    def __init__(self, out_dims, link_type = 'logit', array_type=jnp.float32):
        assert link_type in ['logit', 'probit']
        super().__init__(out_dims, out_dims, link_type, array_type)

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
    
    def sample_Y(self, rate):
        """
        Bernoulli process

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        return jr.bernoulli(prng_state, rate)

        


class Poisson(CountLikelihood):
    """
    Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity
    yₙ is non-negative integer count data
    """

    def __init__(self, out_dims, tbin, link_type = "log", array_type=jnp.float32):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        assert link_type in ['log', 'softplus']
        super().__init__(out_dims, out_dims, tbin, link_type, array_type)

    def log_likelihood(self, f, y):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        
        Poisson(fₙ) = μʸ exp(-μ) / yₙ!
        log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        
        :param y: observed data (yₙ) (out_dims,)
        :param f: latent function value (fₙ) (out_dims,)
        :return:
            log likelihood
        """
        mu = np.maximum(self.inverse_link(f), 1e-8) * self.tbin
        ll = y * np.log(mu) - mu - gammaln(y + 1)
        return ll

    def variational_expectation(
        self, prng_state, y, f_mean, f_cov, jitter, approx_int_method, num_approx_pts, 
    ):
        """
        Closed form of the expected log likelihood for exponential link function
        """
        if self.link_type == 'log':  # closed form for E[log p(y|f)]
            f_var = jnp.diag(f_cov)  # diagonalize
            mu_mean = jnp.maximum(self.link_fn(f_mean), 1e-8)
            ll = y * jnp.log(mu_mean) - mu_mean - gammaln(y + 1)
            
            E_log_lik = -0.5 * (
                _log_twopi
                + (f_var + (y - f_mean) ** 2) / obs_var
                + jnp.log(np.maximum(obs_var, 1e-8))
            )  # (out_dims)
            
            log_lik = log_lik.sum()  # sum over out_dims
            return E_log_lik, dlambda_1, dlambda_2

        else:
            return super().variational_expectation(
                prng_state, y, f_mean, f_cov, jitter, approx_int_method, num_approx_pts, 
            )

    def sample_Y(self, prng_state, rate):
        """
        Bernoulli process

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        mean = rate * self.tbin
        return jr.poisson(prng_state, mean)
    
#             if derivatives:
#                 # dE_dm: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. mₙ of q(f)
#                 # dE_dV: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. Vₙ of q(f)
#                 dEll_dm = (y - f_mean) / obs_var
#                 dEll_dV = -0.5 / obs_var

#                 if mask is not None:
#                     dEll_dm = np.where(mask, 0.0, dEll_dm)  # (out_dims, approx_points)
#                     dEll_dV = np.where(mask, 0.0, dEll_dV)  # (out_dims, approx_points)

#                 dlambda_1 = (dEll_dm - 2 * dEll_dV * f_mean)[:, None]  # (f_dims, 1)
#                 dlambda_2 = np.diag(dEll_dV)  # (f_dims, f_dims)

#             else:
#                 dlambda_1, dlambda_2 = None, None




class NegativeBinomial(CountLikelihood):
    """
    Gamma-Poisson mixture. Negative Binomial likelihood:
        p(yₙ|fₙ;rₙ) = 
    
    Poisson case when r = infty
    
    :param np.ndarray r_inv: :math:`r^{-1}` parameter of the NB likelihood, if left to None this value is 
                           expected to be provided by the heteroscedastic model in the inference class.
    """
    r_inv: Union[jnp.ndarray, None]

    def __init__(self, out_dims, tbin, link_type="log", array_type=jnp.float32):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        assert link_type in ['log', 'softplus']
        super().__init__(out_dims, out_dims, tbin, link_type, array_type)
        self.r_inv = self._to_jax(r_inv) if r_inv is not None else None
        
    def apply_constraints(self):
        if self.r_inv is not None:
            def update(r_inv):
                return jnp.maximum(r_inv, 0.)

            model = jax.tree_map(lambda p: p, self)  # copy
            model = eqx.tree_at(
                lambda tree: tree.r_inv,
                model,
                replace_fn=update,
            )

        return model
            

    def log_likelihood(self, f, y):
        """
        The negative log likelihood function. Note that if disper_param is not None, it will use those values for 
        the dispersion parameter rather than its own dispersion parameters.
        
        .. math::
            P(n|\lambda, r) = \frac{\lambda^n}{n!} \frac{\Gamma(r+n)}{\Gamma(r) \, (r+\lamba)^n} \left( 1 + \frac{\lambda}{r} \right)^{-r}
        
        where the mean is related to the conventional parameter :math:`\lambda = \frac{pr}{1-p}`
        
        For :math:`r \to \infty` we compute the likelihood as a correction to Poisson retaining :math:`\log{1 + O(r^{-1})}`.
        We parameterize the likelihood with :math:`r^{-1}`, as this allows one to reach the Poisson limit (:math:`r^{-1} \to 0`) 
        using the series expansion.
        
        :param int b: batch index to evaluate
        :param jnp.ndarray rates: rates of shape (trial, neuron, time)
        :param jnp.ndarray n_l_rates: spikes*log(rates)
        :param jnp.ndarray spikes: spike counts of shape (trial, neuron, time)
        :param list neuron: list of neuron indices to evaluate
        :param jnp.ndarray disper_param: input for heteroscedastic NB likelihood of shape (trial, neuron, time), 
                                          otherwise uses fixed :math:`r_{inv}`
        :returns: NLL of shape (trial, time)
        :rtype: jnp.ndarray
        """    
        # when r becomes very large, parameterization in r becomes numerically unstable
        asymptotic_mask = (disper_param < 1e-3)
        r_ = 1./(disper_param + asymptotic_mask)
        r_inv_ = disper_param # use 1/r parameterization
        
        #if self.strict_likelihood:
        
        self.totspik.append(spikes.sum(-1))
        self.tfact.append(spikes*torch.log(self.tbin.cpu()))
        self.lfact.append(torch.lgamma(spikes+1.))
            
        tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
        lfact = self.lfact[b][:, neuron, :].to(self.tbin.device)
        if tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]: # cannot rely on broadcasting
            tfact = tfact.repeat(spikes.shape[0]//tfact.shape[0], 1, 1)
            lfact = lfact.repeat(spikes.shape[0]//lfact.shape[0], 1, 1)
        #else:
        #    tfact, lfact = 0, 0
    
        lambd = rates*self.tbin
        fac_lgamma = (jnp.lgamma(r_+spikes) - jnp.lgamma(r_))
        fac_power = (-(spikes+r_)*jnp.log(r_+lambd) + r_*jnp.log(r_))
        
        ll_r = fac_power + fac_lgamma
        ll_r_inv = -lambd - jnp.log(1. + r_inv_*(spikes**2 + 1. - spikes*(3/2 + lambd)))
        
        ll = -log_rates + tfact - lfact
        ll[asymptotic_mask] = ll[asymptotic_mask] + ll_r_inv[asymptotic_mask]
        ll[~asymptotic_mask] = ll[~asymptotic_mask] + ll_r[~asymptotic_mask]
        #nll = nll_r*(~asymptotic_mask) + nll_r_inv*asymptotic_mask
        return nll.sum(1)

    def sample_Y(self, rate, r_inv=None):
        """
        Sample from the Gamma-Poisson mixture.
        
        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :param int max_count: maximum number of spike counts per time bin
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        if r_inv is not None:
            r_ = 1./(self.r_inv[None, :, None].expand(rate.shape[0], self.neurons, 
                                                      rate_.shape[-1]).data.cpu().numpy()
                     + 1e-12)[:, neuron, :]
        else:
            samples = rate.shape[0]
            with jnp.no_grad():
                disp = self.sample_dispersion(XZ, rate.shape[0]//XZ.shape[0], neuron)
            r_ = 1./(disp.cpu().numpy() + 1e-12)
            
        s = jr.gamma(r_, rate_*self.tbin.item()/r_) # becomes delta around rate*tbin when r to infinity, cap at 1e12
        return jr.poisson(jnp.array(s)).numpy()

    
    


class ZeroInflatedPoisson(CountLikelihood):
    """
    Zero-inflated Poisson (ZIP) count likelihood. [1]
    
    References:
    
    [1] `Untethered firing fields and intermittent silences: Why grid‐cell discharge is so variable`, 
        Johannes Nagele  Andreas V.M. Herz  Martin B. Stemmler
    
    """
    
    arctanh_alpha: Union[jnp.ndarray, None]

    def __init__(self, out_dims, tbin, alpha, link_type="log"):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(out_dims, out_dims, tbin, link_type, array_type)
        self.arctanh_alpha = jnp.arctanh(self._to_jax(alpha)) if alpha is not None else None

    def log_likelihood(self, f, y):
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
        if disper_param is None:
            alpha_ = jnp.tanh(self.arctanh_alpha)(1, self.neurons)[:, neuron, None]
        else:
            alpha_ = disper_param
            
        tfact, lfact = self.get_saved_factors(b, neuron, spikes)
        T = rates*self.tbin
        zero_spikes = (spikes == 0) # mask
        nll_ = (-n_l_rates + T - tfact + lfact - jnp.log(1.-alpha_)) # -log (1-alpha)*p(N)
        p = jnp.exp(-nll_) # stable as nll > 0
        nll_0 = -jnp.log(alpha_ + p)
        nll = zero_spikes*nll_0 + (~zero_spikes)*nll_
        return nll.sum(1)
 
            
    def apply_constraints(self):
        if self.arctanh_alpha is not None:
            def update(arctanh_alpha):
                return jnp.maximum(arctanh_alpha, 0.)

            model = jax.tree_map(lambda p: p, self)  # copy
            model = eqx.tree_at(
                lambda tree: tree.r_inv,
                model,
                replace_fn=update,
            )
            return model
        
        else:
            return self
    
    def sample_Y(self, rate, neuron=None, XZ=None):
        """
        Sample from ZIP process.
        
        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        rate_ = rate[:, neuron, :]
        
        if self.dispersion_mapping is None:
            alpha_ = self.alpha[None, :, None].expand(rate.shape[0], self.neurons, 
                                                      rate_.shape[-1]).data.cpu().numpy()[:, neuron, :]
        else:
            with jnp.no_grad():
                alpha_ = self.sample_dispersion(XZ, rate.shape[0]//XZ.shape[0], neuron).cpu().numpy()
            
        zero_mask = point_process.gen_IBP(alpha_)
        return (1.-zero_mask)*jnp.poisson(jnp.array(rate_*self.tbin.item())).numpy()



class ConwayMaxwellPoisson(CountLikelihood):
    """
    Conway-Maxwell-Poisson, as described in
    https://en.wikipedia.org/wiki/Conway%E2%80%93Maxwell%E2%80%93Poisson_distribution.

    """
    log_nu: Union[jnp.ndarray, None]
    powers: jnp.ndarray
    jfact: jnp.ndarray

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        nu,
        J=100,
        array_type=jnp.float32,
    ):
        """
        :param int J: number of terms to use for partition function sum
        """
        super().__init__(tbin, neurons, inv_link, array_type)
        self.log_nu = jnp.log(self._to_jax(nu)) if nu is not None else None

        self.powers = jnp.arange(J + 1).astype(self.array_type)
        self.jfact = gammaln(self.powers + 1.0)
        
    def apply_constraints(self):
        if self.log_nu is not None:
            def update(log_nu):
                return jnp.minimum(jnp.maximum(log_nu, -3.), 3.)  # stable range

            model = jax.tree_map(lambda p: p, self)  # copy
            model = eqx.tree_at(
                lambda tree: tree.r_inv,
                model,
                replace_fn=update,
            )
            return model
        
        else:
            return self

    def log_Z(self, log_lambda, nu):
        """
        Partition function.

        :param jnp.ndarray lambd: lambda of shape (samples, neurons, timesteps)
        :param jnp.ndarray nu: nu of shape (samples, neurons, timesteps)
        :returns: log Z of shape (samples, neurons, timesteps)
        :rtype: jnp.ndarray
        """
        # indx = jnp.where((self.powers*lambd.max() - nu_.min()*self.j) < -1e1) # adaptive
        # if len(indx) == 0:
        #    indx = self.J+1
        log_Z_term = (
            self.powers[:, None, None, None] * log_lambda[None, ...]
            - nu[None, ...] * self.jfact[:, None, None, None]
        )
        return jax.nn.logsumexp(log_Z_term, axis=0)

    def nll(self, b, rates, n_l_rates, spikes, neuron, disper_param=None, strict_likelihood=True):
        """
        :param int b: batch index to evaluate
        :param jnp.ndarray rates: rates of shape (trial, neuron, time)
        :param jnp.ndarray n_l_rates: spikes*log(rates)
        :param jnp.ndarray spikes: spike counts of shape (neuron, time)
        :param list neuron: list of neuron indices to evaluate
        :param jnp.ndarray disper_param: input for heteroscedastic NB likelihood of shape (trial, neuron, time),
                                          otherwise uses fixed :math:`\nu`
        :returns: NLL of shape (trial, time)
        :rtype: jnp.ndarray
        """
        if disper_param is None:
            nu_ = jnp.exp(self.log_nu).expand(1, self.neurons)[:, neuron, None]
        else:
            nu_ = jnp.exp(disper_param)  # nn.functional.softplus

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

        log_lambda = jnp.log(rates * self.tbin + 1e-12)

        l_Z = self.log_Z(log_lambda, nu_)

        nll = -n_l_rates + l_Z - tfact + nu_ * lfact
        return nll.sum(1)

    def sample_Y(self, rate, neuron=None, XZ=None):
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
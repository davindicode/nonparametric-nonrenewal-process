from functools import partial

import jax.numpy as np
from jax import jit, jacrev, random, vmap, value_and_grad, grad, tree_map

from jax.scipy.special import erf, gammaln
from jax.scipy.linalg import block_diag, solve_triangular
from jax.numpy.linalg import cholesky
from jax.nn import softmax

from .utils import expsum, softplus, sigmoid, logphi, softplus_inv, gauss_hermite, \
    get_blocks, inv, mc_sample

import math
_log_twopi = math.log(2 * math.pi)







### classes ###
class FactorizedLikelihood(object):
    """
    The likelihood model class, p(yₙ|fₙ)
    
    variational_expectation() computes all E_q(f) related quantities and its derivatives for NGD
    We allow multiple f (vector fₙ) to correspond to a single yₙ as in heteroscedastic likelihoods
    
    The default functions here use cubature/MC approximation methods, exact integration is specific 
    to certain likelihood classes.
    """
    def __init__(self, out_dims, f_dims, hyp):
        """
        all hyperparameters stored are pre-transformation
        :param hyp: (hyper)parameters of the likelihood model
        """
        self.hyp = hyp
        self.f_dims = f_dims
        self.out_dims = out_dims
        self.num_f_per_out = self.f_dims // self.out_dims
        

    @partial(jit, static_argnums=(0, 5))
    def grads_log_likelihood_n(self, f_mean, df_points, y, lik_params, derivatives):
        """
        Factorization over data points n, vmap over out_dims
        vmap over approx_points
        
        :param np.array f_mean: (scalar) or (cubature,) for cubature_dim > 1
        :param np.array df_points: (approx_points,) or (cubature, approx_points) for cubature_dim > 1
        
        :return:
            expected log likelihood ll (approx_points,)
            dll_dm (approx_points,) or (cubature, approx_points)
            d2ll_dm2 (approx_points,) or (cubature, cubature, approx_points)
        """
        f = df_points + f_mean # (approx_points,) or (cubature, approx_points) for cubature_dim > 1
        
        if derivatives:
            def grad_func(f):
                ll, dll_dm = value_and_grad(self.log_likelihood_n, argnums=0)(f, y, lik_params)
                return dll_dm, (ll, dll_dm)

            def temp_func(f):
                #dll_dm, (ll,) = grad_func(f)
                d2ll_dm2, aux = jacrev(grad_func, argnums=0, has_aux=True)(f)
                ll, dll_dm = aux
                return ll, dll_dm, d2ll_dm2
            
            ll, dll_dm, d2ll_dm2 = vmap(temp_func, in_axes=0, out_axes=(0, 0, 0))(f)
        
        else:
            ll = vmap(self.log_likelihood_n, (0, None, None))(f, y, lik_params)
            dll_dm, d2ll_dm2 = None, None
        
        return ll, dll_dm, d2ll_dm2
        
    def log_likelihood_n(self, f, y, lik_params):
        raise NotImplementedError('direct evaluation of this log-likelihood is not implemented')

    @staticmethod
    def link_fn(latent_mean):
        return latent_mean
    
    @partial(jit, static_argnums=(0, 8))
    def variational_expectation(self, lik_params, prng_state, jitter, y, mask, f_mean, f_cov, derivatives=False):
        """
        E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ and its derivatives
        The log marginal likelihood is log E[p(yₙ|fₙ)] = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ 
        onlt the block-diagonal part of f_std matters
        :param np.array f_mean: mean of q(f) of shape (f_dims,)
        :param np.array f_cov: covariance of q(f) of shape (f_dims, f_dims)
        :return:
            log likelihood: expected log likelihood
            dlambda_1: gradient of E_q(f)[ log p(y|f) ] w.r.t. mean natural parameter
            dlambda_2: gradient of E_q(f)[ log p(y|f) ] w.r.t. covariance natural parameter
        """
        cubature_dim = self.num_f_per_out # use smaller subgrid for cubature
        f, w = self.approx_int_func(cubature_dim, prng_state)
        
        ### compute transformed f locations ###
        # turn f_cov into lower triangular block diagonal matrix f_
        if cubature_dim == 1:
            f = np.tile(f, (self.out_dims, 1)) # copy over out_dims, (out_dims, cubature_dim)
            f_var = np.diag(f_cov)
            f_std = np.sqrt(f_var)
            f_mean = f_mean[:, None] # (f_dims, 1)
            df_points = f_std[:, None] * f # (out_dims, approx_points)
            
        else: # block-diagonal form
            f = np.tile(f[None, ...], (self.out_dims, 1, 1)) # copy subgrid (out_dims, cubature_dim, approx_points)
            f_cov = get_blocks(np.diag(np.diag(f_cov)), self.out_dims, cubature_dim)
            #chol_f_cov = np.sqrt(np.maximum(f_cov, 1e-12)) # diagonal, more stable
            chol_f_cov = cholesky(f_cov + jitter*np.eye(cubature_dim)[None, ...]) # (out_dims, cubature_dim, cubature_dim)
            
            f_mean = f_mean.reshape(self.out_dims, cubature_dim, 1)
            df_points = (chol_f_cov @ f) # (out_dims, cubature_dim, approx_points)
           
        ### derivatives ###
        in_shape = tree_map(lambda x: 0, lik_params)
        if derivatives:
            ll, dll_dm, d2ll_dm2 = vmap(self.grads_log_likelihood_n, 
                                        in_axes=(0, 0, 0, in_shape, None), 
                                        out_axes=(0, 0, 0))(
                f_mean, df_points, y, lik_params, True) # vmap over out_dims
            
            if mask is not None: # apply mask
                dll_dm = np.where(mask[:, None], 0., dll_dm) # (out_dims, approx_points)
                d2ll_dm2 = np.where(mask[:, None], 0., d2ll_dm2) # (out_dims, approx_points)
            
            dEll_dm = (w[None, :] * dll_dm).sum(1)
            d2Ell_dm2 = (w[None, :] * d2ll_dm2).sum(1)

            if cubature_dim == 1: # only need diagonal f_cov
                dEll_dV = .5 * d2Ell_dm2
                dlambda_1 = (dEll_dm - 2 * (dEll_dV * f_mean[:, 0]))[:, None] # (f_dims, 1)
                dlambda_2 = np.diag(dEll_dV) # (f_dims, f_dims)

            else:
                dEll_dV = .5 * d2Ell_dm2[..., 0]
                dlambda_1 = (dEll_dm[:, None] - 2 * (dEll_dV @ f_mean).reshape(-1, 1)) # (f_dims, 1)
                dlambda_2 = dEll_dV # (f_dims, f_dims)
            
        else: # only compute log likelihood
            ll, dll_dm, d2ll_dm2 = vmap(self.grads_log_likelihood_n, 
                                        in_axes=(0, 0, 0, in_shape, None), 
                                        out_axes=(0, 0, 0))(
                f_mean, df_points, y, lik_params, False) # vmap over n
            dlambda_1, dlambda_2 = None, None
            
        ### expected log likelihood ###
        # f_mean and f_cov are from P_smoother
        if mask is not None: # apply mask
            ll = np.where(mask[:, None], 0., ll) # (out_dims, approx_points)
        weighted_log_lik = w * ll.sum(0) # (approx_pts,)
        E_log_lik = weighted_log_lik.sum() # E_q(f)[log p(y|f)]
        
        return E_log_lik, dlambda_1, dlambda_2
        
    ### evaluate ###
    def posterior_predictive(self, f_mean, f_cov):
        """
        Given posterior of q(f), compute q(a) for likelihood variables
        """
        raise NotImplementedError('Posterior evaluation of likelihood variables not implemented')
    
    def posterior_sample(self, f_samp):
        """
        Sample from posterior
        """
        raise NotImplementedError('Posterior sampling of likelihood variables not implemented')



### density likelihoods ###
class Gaussian(FactorizedLikelihood):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    """
    def __init__(self, out_dims, variance):
        """
        :param variance: The observation noise variance, σ²
        """
        super().__init__(out_dims, out_dims, hyp={'variance': variance})

    @property
    def variance(self):
        return softplus(self.hyp.values()[0])

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, f, y, hyp=None):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q approximation/cubature points.
        
        :param y: observed data yₙ [out_dims, 1]
        :param f: mean, i.e. the latent function value fₙ [out_dims, Q]
        :param hyp: likelihood variance σ² [scalar]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise [out_dims, Q]
        """
        hyp = self.hyp if params is None else params
        obs_var = np.maximum(softplus(hyp['variance']), 1e-8)
        
        ll = jax.vmap(jax.scipy.stats.norm.logpdf, in_axes=(None, 1, None), out_axes=1)(f, y, obs_var)
        #var = var[:, None]
        #ll = -.5 * (_log_twopi * np.log(var) + (y - f)**2 / var)
        return ll

    @partial(jit, static_argnums=(0, 8))
    def variational_expectation(self, lik_params, prng_state, jitter, y, mask, f_mean, f_cov, derivatives=True):
        """
        Exact, ignore approx_int_func
        
        log Zₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
        
        ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[log 𝓝(yₙ|fₙ,σ²)]
        
        :param np.array f_mean: q(f) mean with shape (f_dims,)
        :param np.array f_cov: q(f) mean with shape (f_dims, f_dims)
        """
        hyp = self.hyp if lik_params is None else lik_params
        obs_var = np.maximum(softplus(hyp['variance']), 1e-8)
        f_var = np.diag(f_cov) # diagonalize
        
        if derivatives:
            # dE_dm: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. mₙ of q(f)
            # dE_dV: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. Vₙ of q(f)
            dEll_dm = (y - f_mean) / obs_var
            dEll_dV = -0.5 / obs_var
            
            if mask is not None:
                dEll_dm = np.where(mask, 0., dEll_dm) # (out_dims,)
                dEll_dV = np.where(mask, 0., dEll_dV) # (out_dims,)
            
            dlambda_1 = (dEll_dm - 2 * dEll_dV * f_mean)[:, None] # (f_dims, 1)
            dlambda_2 = np.diag(dEll_dV) # (f_dims, f_dims)
            
        else:
            dlambda_1, dlambda_2 = None, None
              
        # ELL
        log_lik = -0.5*(
            _log_twopi + (f_var + (y - f_mean)**2) / obs_var + 
            np.log(obs_var)
        ) # (out_dims)
        
        # apply mask
        #mask = np.isnan(y)
        if mask is not None:
            log_lik = np.where(mask, 0., log_lik) # (out_dims,)
        log_lik = log_lik.sum() # sum over out_dims
        return log_lik, dlambda_1, dlambda_2
      
    

class HeteroscedasticGaussian(FactorizedLikelihood):
    """
    Heteroscedastic Gaussian likelihood 
        p(y|f1,f2) = N(y|f1,link(f2)^2)
    """
    def __init__(self, out_dims, link='softplus'):
        """
        :param link: link function, either 'exp' or 'softplus' (note that the link is modified with an offset)
        """
        super().__init__(out_dims, 2*out_dims, None)
        if link == 'exp':
            self.link_fn = lambda x: np.exp(x)
            self.dlink_fn = lambda x: np.exp(x)
        elif link == 'softplus':
            self.link_fn = lambda x: softplus(x)
            self.dlink_fn = lambda x: sigmoid(x)
        else:
            raise NotImplementedError('link function not implemented')

    @partial(jit, static_argnums=(0,))
    def log_likelihood_n(self, f, y, hyp):
        """
        Evaluate the log-likelihood
        :return:
            log likelihood of shape (approx_points,)
        """
        mu, var = f[0], np.maximum(self.link_fn(f[1])**2, 1e-8)
        ll = -0.5 * ( (_log_twopi + np.log(var)) + (y - mu)**2 / var)
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
        if link == 'logit':
            self.link_fn = lambda f: 1 / (1 + np.exp(-f))
            self.dlink_fn = lambda f: np.exp(f) / (1 + np.exp(f)) ** 2
            
        elif link == 'probit':
            jitter = 1e-8
            self.link_fn = lambda f: 0.5 * (1.0 + erf(f / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter
            self.dlink_fn = lambda f: grad(self.link_fn)(np.squeeze(f)).reshape(-1, 1)
            
        else:
            raise NotImplementedError('link function not implemented')

        
    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, f, y, hyp=None):
        """
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ ϵ ℝ
        :param hyp: dummy input, Probit has no hyperparameters
        :return:
            log p(yₙ|fₙ), p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾
        """
        #return np.where(np.equal(y, 1), self.link_fn(f), 1 - self.link_fn(f))
        return np.where(np.equal(y, 1), np.log(self.link_fn(f)), np.log(1 - self.link_fn(f)))




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
    def __init__(self, out_dims, tbin, link='exp'):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(out_dims, out_dims, tbin, None)
        self.tbin = tbin
        self.link = link
        if link == 'exp':
            self.link_fn = lambda mu: np.exp(mu)
            self.dlink_fn = lambda mu: np.exp(mu)
        elif link == 'softplus':
            self.link_fn = lambda x: softplus(x)
            self.dlink_fn = lambda x: sigmoid(x)
        else:
            raise NotImplementedError('link function not implemented')
            

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
        mu = np.maximum(self.link_fn(f), 1e-8)*self.tbin
        ll = (y * np.log(mu) - mu - gammaln(y + 1))
        return ll

    
    @partial(jit, static_argnums=(0, 8))
    def variational_expectation(self, lik_params, prng_state, jitter, y, mask, f_mean, f_cov, derivatives=True):
        """
        Closed form of the expected log likelihood for exponential link function
        """
        if False:#derivatives and self.link == 'exp':  # closed form for E[log p(y|f)]
            hyp = self.hyp if lik_params is None else lik_params
            f_var = np.diag(f_cov) # diagonalize
            mu_mean = np.maximum(self.link_fn(f_mean), 1e-8)
            ll = (y * np.log(mu_mean) - mu_mean  - gammaln(y + 1))

            if derivatives:
                # dE_dm: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. mₙ of q(f)
                # dE_dV: derivative of E_q(f)[log p(yₙ|fₙ)] w.r.t. Vₙ of q(f)
                dEll_dm = (y - f_mean) / obs_var
                dEll_dV = -0.5 / obs_var

                if mask is not None:
                    dEll_dm = np.where(mask, 0., dEll_dm) # (out_dims, approx_points)
                    dEll_dV = np.where(mask, 0., dEll_dV) # (out_dims, approx_points)

                dlambda_1 = (dEll_dm - 2 * dEll_dV * f_mean)[:, None] # (f_dims, 1)
                dlambda_2 = np.diag(dEll_dV) # (f_dims, f_dims)

            else:
                dlambda_1, dlambda_2 = None, None

            # ELL
            E_log_lik = -0.5*(
                _log_twopi + (f_var + (y - f_mean)**2) / obs_var + 
                np.log(np.maximum(obs_var, 1e-8))
            ) # (out_dims)

            # apply mask
            #mask = np.isnan(y)
            if mask is not None:
                log_lik = np.where(mask, 0., log_lik) # (out_dims,)
            log_lik = log_lik.sum() # sum over out_dims
            return E_log_lik, dlambda_1, dlambda_2
                
                
        else:
            return super().variational_expectation(
                lik_params, prng_state, jitter, y, mask, f_mean, f_cov, derivatives)



class NegativeBinomial(CountLikelihood):
    """
    NB likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    """
    def __init__(self, out_dims, tbin, link='exp'):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(out_dims, out_dims, tbin, hyp)
        self.link = link
        if link == 'exp':
            self.link_fn = lambda mu: np.exp(mu)
            self.dlink_fn = lambda mu: np.exp(mu)
        elif link == 'logistic':
            self.link_fn = lambda mu: softplus(mu)
            self.dlink_fn = lambda mu: sigmoid(mu)
        else:
            raise NotImplementedError('link function not implemented')
            
            
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
        mu = np.maximum(self.link_fn(f), 1e-8)*self.tbin
        ll = (y * np.log(mu) - mu - gammaln(y + 1))
        return ll
            
            
            
class ZeroInflatedPoisson(CountLikelihood):
    """
    ZIP likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    """
    def __init__(self, out_dims, tbin, link='exp'):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(out_dims, out_dims, tbin, hyp)
        self.link = link
        if link == 'exp':
            self.link_fn = lambda mu: np.exp(mu)
            self.dlink_fn = lambda mu: np.exp(mu)
        elif link == 'logistic':
            self.link_fn = lambda mu: softplus(mu)
            self.dlink_fn = lambda mu: sigmoid(mu)
        else:
            raise NotImplementedError('link function not implemented')
            
            
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
        mu = np.maximum(self.link_fn(f), 1e-8)*self.tbin
        ll = (y * np.log(mu) - mu - gammaln(y + 1))
        return ll
            
            
            
            
class UniversalCount(CountLikelihood):
    """
    Universal count likelihood
    """
    def __init__(self, out_dims, C, K, tbin):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(out_dims, C*out_dims, tbin, hyp)
        self.K = K
            
            
    @partial(jit, static_argnums=(0,))
    def log_likelihood_n(self, f, y, hyp=None):
        """
        Evaluate the softmax from 0 to K with linear mapping from f
        """
        hyp = self.hyp if params is None else params
        W = hyp['W']
        b = hyp['b']
        logp_cnts = log_softmax(W @ f + b)
        return logp_cnts[int(y)]
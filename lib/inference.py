from typing import Union

import math
import numpy as np

from functools import partial

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr


from .base import module
from .likelihoods.base import FactorizedLikelihood, RenewalLikelihood
from .likelihoods.factorized import LogCoxProcess
from .GP.base import GP
from .GP.markovian import GaussianLTI, MultiOutputLTI
from .GP.spatiotemporal import KroneckerLTI
from .filters.base import Filter

_log_twopi = math.log(2 * math.pi)



### base ###
class FilterGLM(module):
    """
    The input-ouput (IO) mapping is deterministic or stochastic
    
    Sample from prior, then use cubature for E_q(f|x) log p(y|f)
    
    Examples: GPFA, GLM (deterministic), GPLVM (stochastic)
    
    Spiketrain filter + GLM with optional SSGP latent states
    """
    
    ssgp: Union[GaussianLTI, None]
    spikefilter: Union[Filter, None]
        
    def __init__(self, ssgp, spikefilter, array_type):
        # checks
        if ssgp is not None:
            assert ssgp.array_type == array_type
            
        if spikefilter is not None:
            assert spikefilter.array_type == array_type
            
        super().__init__(array_type)
        self.ssgp = ssgp
        self.spikefilter = spikefilter
        
    
    
    
class FilterGPLVM(module):
    """
    Spiketrain filter + GP with optional SSGP latent states
    """
    
    ssgp: Union[GaussianLTI, None]
    spikefilter: Union[Filter, None]
        
    def __init__(self, ssgp, spikefilter, array_type):
        # checks
        if ssgp is not None:
            assert ssgp.array_type == array_type
            
        if spikefilter is not None:
            assert spikefilter.array_type == array_type
            
        super().__init__(array_type)
        self.ssgp = ssgp
        self.spikefilter = spikefilter
        
        
    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        def update(shape):
            return jnp.minimum(jnp.maximum(shape, 1e-5), 2.5)
        
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.ssgp,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )
        
        model = eqx.tree_at(
            lambda tree: tree.spikefilter,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model
        
        
    def _sample_input_trajectories(self, x, t, num_samps):
        """
        Combines observed inputs with latent trajectories
        """
        if self.ssgp is not None:
            x_samples, KL_ss = self.ssgp.sample_posterior(
                ss_params,
                ss_var_params,
                prng_keys[0],
                num_samps,
                timedata,
                None,
                jitter,
                compute_KL=True,
            )  # (time, tr, x_dims, 1)
        
        return inputs, KL
    
    
    def _sample_input_marginals(self, x, t, num_samps):
        """
        Combines observed inputs with latent marginal samples
        """
        if self.ssgp is not None:  # filtering-smoothing
            x_samples, KL_ss = self.ssgp.evaluate_posterior(
                ss_params,
                ss_var_params,
                prng_keys[0],
                num_samps,
                timedata,
                None,
                jitter,
                compute_KL=True,
            )  # (time, tr, x_dims, 1)
        
        return inputs, KL
    
    
    def ELBO(self, prng_state, x, t, num_samps):
        raise NotImplementedError
        
    

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        """
        spike coupling filter
        """
        batch_edge, _, _ = self.batch_info
        spk = self.all_spikes[
            :, neuron, batch_edge[b] : batch_edge[b + 1] + self.history_len
        ].to(self.likelihood.dt.device)
        spk_filt, spk_var = self.filter(spk, XZ)  # trials, neurons, timesteps
        mean = F_mu + spk_filt
        variance = F_var + spk_var
        return self.likelihood.objective(mean, variance, XZ, b, neuron, samples, mode)

    def filtered_rate(self, F_mu, F_var, unobs_neuron, trials, MC_samples=1):
        """
        Evaluate the instantaneous rate after spike coupling, with unobserved neurons not contributing
        to the filtered population rate.
        """
        unobs_neuron = self.likelihood._validate_neuron(unobs_neuron)
        batch_edge, _, _ = self.batch_info
        spk = self.all_spikes[
            :, neuron, batch_edge[b] : batch_edge[b + 1] + self.history_len
        ].to(self.likelihood.dt.device)

        with torch.no_grad():
            hist, hist_var = self.spike_filter(spk, XZ)
            hist[:, unobs_neuron, :] = 0  # mask
            hist_var[:, unobs_neuron, :] = 0  # mask
            h = self.mc_gen(
                F_mu + hist, F_var + hist_var, MC_samples, torch.arange(self.neurons)
            )
            intensity = self.likelihood.f(h.view(-1, trials, *h.shape[1:]))

        return intensity

    def sample(self, F_mu, ini_train, neuron=None, XZ=None, obs_spktrn=None):
        """
        Assumes all neurons outside neuron are observed for spike filtering.

        :param torch.Tensor F_mu: input F values of shape (MC, trials, neurons, time)
        :param np.ndarray ini_train: initial spike train of shape (trials, neurons, time)
        :returns: spike train and instantaneous firing rates of shape (trials, neurons, time)
        :rtype: tuple of np.array
        """
        neuron = self.likelihood._validate_neuron(neuron)
        n_ = list(range(self.likelihood.neurons))

        MC, trials, N, steps = F_mu.shape
        if trials != ini_train.shape[0] or N != ini_train.shape[1]:
            raise ValueError("Initial spike train shape must match input F tensor.")
        spikes = []
        spiketrain = torch.empty(
            (*ini_train.shape[:2], self.history_len), device=self.likelihood.dt.device
        )

        iterator = tqdm(range(steps), leave=False)  # AR sampling
        rate = []
        for t in iterator:
            if t == 0:
                spiketrain[..., :-1] = torch.tensor(
                    ini_train, device=self.likelihood.dt.device
                )
            else:
                spiketrain[..., :-2] = spiketrain[..., 1:-1].clone()  # shift in time
                spiketrain[..., -2] = torch.tensor(
                    spikes[-1], device=self.likelihood.dt.device
                )

            with torch.no_grad():  # spiketrain last time element is dummy, [:-1] used
                if XZ is None:
                    cov_ = None
                else:
                    cov_ = XZ[:, t : t + self.history_len, :]
                hist, hist_var = self.filter(spiketrain, cov_)

            rate_ = (
                self.likelihood.f(F_mu[..., t] + hist[None, ..., 0])
                .mean(0)
                .cpu()
                .numpy()
            )  # (trials, neuron)

            if obs_spktrn is None:
                spikes.append(
                    self.likelihood.sample(rate_[..., None], n_, XZ=XZ)[..., 0]
                )
                # spikes.append(point_process.gen_IBP(1. - np.exp(-rate_*self.likelihood.dt.item())))
            else:  # condition on observed spike train partially
                spikes.append(obs_spktrn[..., t])
                spikes[-1][:, neuron] = self.likelihood.sample(
                    rate_[..., None], neuron, XZ=XZ
                )[..., 0]
                # spikes[-1][:, neuron] = point_process.gen_IBP(1. - np.exp(-rate_[:, neuron]*self.likelihood.dt.item()))
            rate.append(rate_)

        rate = np.stack(rate, axis=-1)  # trials, neurons, timesteps
        spktrain = np.transpose(
            np.array(spikes), (1, 2, 0)
        )  # trials, neurons, timesteps

        return spktrain, rate
    
    
    
    
    
class FactorizedGPLVM(GLMGPLVM):
    """
    Factorization across time points allows one to rely on latent marginals
    """

    gp: GP
    likelihood: FactorizedLikelihood
        
    def __init__(self, gp, ssgp = None, spikefilter = None):
        # checks
        assert likelihood.array_type == gp.array_type
        assert likelihood.f_dims == gp.kernel.out_dims
        
        super().__init__(ssgp, spikefilter, gp.array_type)
        self.gp = gp
        self.likelihood = likelihood
        
        self.x_dims = state_space.kernel.out_dims
        self.state_space = state_space

        self.y = None  # no training data set

    ### variational inference ###
    def ELBO(
        self,
        prng_state,
        num_samps,
    ):
        """
        Compute ELBO
        """
        xx = np.linspace(-5., 5., 100)[None, None, :, None]

        qf_m, qf_c, _, _ = svgp.evaluate_posterior(
            xx, mean_only=False, diag_cov=False, compute_KL=False, compute_aux=False, jitter=jitter)
        
        # likelihood
        self.likelihood

        return objective, grads
    

    ### sample ###
    def sample_prior(self, prng_state, num_samps, x_obs=None, timedata=None):
        """
        Sample from the generative model
        """
        pf_x = self.gp.sample_prior(prng_state, xx.repeat(num_samps, axis=0), jitter)  # (evals, samp, f_dim)
        return y, q_vh, I, eps_samples

    def sample_posterior(self, prng_state, num_samps):
        """
        Sample from posterior predictive
        """
        qf_x, KL = self.gp.sample_posterior(
            prng_state, xx.repeat(num_samps, axis=0), jitter, compute_KL=True) # (evals, samp, f_dim)

        return y, q_vh, I, eps_samples

    
    ### evaluation ###
    def evaluate_metric(self):
        return
    
    
    
class RenewalGPLVM(module):
    """
    Renewal likelihood GPLVM
    """
    
    gp: GP
    likelihood: RenewalLikelihood
    
    def __init__(self, gp, likelihood):
        # checks
        assert likelihood.array_type == gp.array_type
        assert likelihood.f_dims == gp.kernel.out_dims
        
        super().__init__(gp.array_type)
        self.gp = gp
        self.likelihood = likelihood
        
    def _gp_sample(self, prng_state, x_eval, prior, jitter):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        
        :param jnp.ndarray x_eval: evaluation locs (num_samps, out_dims, eval_locs, in_dims)
        """
        if prior:
            pre_rates = svgp.sample_prior(prng_state, x_eval, jitter)  # (evals, samp, f_dim)

        else:
            pre_rates, _ = svgp.sample_posterior(
                prng_state, x_eval, jitter, compute_KL=False) # (evals, samp, f_dim)

        return pre_rates
        
    ### inference ###
    def ELBO(self):
        qf_x, KL = svgp.sample_posterior(prng_state, xx.repeat(num_samps, axis=0), jitter, compute_KL=True) # (evals, samp, f_dim)
        return
        
    
    ### evaluation ###    
    def sample_conditional_intensity(self, prng_state, tx_eval):
        """
        rho(t) = r(t) p(u) / (1 - int p(u) du)
        """
        pre_rates = self._gp_sample(prng_state, x_eval, prior, jitter)
        
        rates = self.link_fn(pre_rates)
        taus = self.dt * jnp.cumsum(rates, axis=2)
        
        # rate rescaling
        rISI = jnp.empty((mc, self.out_dims, num_ISIs))
        
        for en, spkinds in enumerate(spiketimes):
            isi_count = jnp.maximum(spkinds.shape[0] - 1, 0)
            
            def body(i, val):
                val[:, en, i] = taus[:, i]
                return val
            
            rISI[:, en, :] = lax.fori_loop(0, isi_count, body, rISI[:, en, :])
            
        return rhot  # (num_samps, out_dims, ts)
    
    def sample_instantaneous_renewal(self, prng_state, t_eval, x_cond):
        """
        :param jnp.ndarray t_eval: evaluation times since last spike, i.e. ISI
        """
        
        return

    
        
        
class NonparametricPPGPLVM(module):
    """
    Bayesian nonparametric modulated point process likelihood
    """
    
    gp: Union[MultiOutputLTI, KroneckerLTI]
    likelihood: LogCoxProcess
        
    t0: jnp.ndarray
    refract_tau: jnp.ndarray
    mean_bias: jnp.ndarray
    
    
    def __init__(self, gp, t0, refract_tau, mean_bias, dt):
        """
        :param jnp.ndarray t0: time transform timescales of shape (out_dims,)
        :param jnp.ndarray tau: refractory mean timescales of shape (out_dims,)
        """            
        super().__init__(t0, refract_tau, mean_bias, dt, gp.array_type)
        self.gp = gp
        self.likelihood = LogCoxProcess(gp.kernel.out_dims, dt, self.array_type)
        
        self.t0 = t0
        self.refract_tau = refract_tau
        self.mean_bias = mean_bias
        
        self.modulated = False if type(gp) == MultiOutputLTI else True
     
    ### functions ###
    def log_time_transform(self, t, inverse=False):
        """
        Inverse transform is from tau [0, 1] to t in R
        """
        if inverse:
            t_ = -jnp.log(1 - t) * self.t0
            log_jac = jnp.log(self.t0) - jnp.log(1 - t) # t0 / (1 - t)
        else:
            s = jnp.exp(-t / self.t0)
            t_ = 1 - s
            log_jac = -t / self.t0 - jnp.log(self.t0)  # s / t0
        return t_, log_jac


    def refractory_mean(self, tau, neg=-12.):
        """
        Refractory period implemented by GP mean function
        """
        return neg * jnp.exp(-tau / self.refract_tau) + self.mean_bias
        
        
    def _transform_gp_sample(self, prng_state, tau_eval, x_eval, prior):
        """
        Obtain the log conditional intensity given input path along which to evaluate
        """
        if self.modulated:
            return
        
        else:
            if prior:
                f_samples = self.gp.sample_prior(
                    prng_state, num_samps, tau_eval, jitter)

            else:
                f_samples, _ = self.gp.sample_posterior(
                    prng_state, num_samps, tau_eval, jitter, False)  # (tr, time, N, 1)
            
        m_eval = self.refractory_mean(tau_eval)
        log_rhotau = f_samples[..., 0] + m_eval[None, :, None]
        return log_rhotau
        
        
    ### inference ###
    def ELBO(self, prng_state, x, t, num_samps):
        self.gp.evaluate_posterior()
        
        self.likelihood.variational_expectation()
        
        return
    
        
    ### sampling ###
    def sample_conditional_intensity(self, prng_state, t_eval):
        """
        Evaluate the conditional intensity along an input path
        """
        tau_eval, log_dtau_dt_eval = self.log_time_transform(t_eval, inverse=False)

        log_rhotau_eval = self._transform_gp_sample(prng_state, tau_eval, x_eval, prior)
        log_rhot_eval = log_rhotau_eval + log_dtau_dt_eval[None, :, None]  
        return jnp.exp(log_rhot_eval)
        
        
    def sample_instantaneous_renewal(self, prng_state, t_eval, x_cond, num_samps = 20, int_eval_pts = 1000, num_quad_pts = 100, prior = True):
        """
        Compute the instantaneous renewal density with rho(ISI;X) from model
        Uses linear interpolation with Gauss-Legendre quadrature for integrals
        
        :param jnp.ndarray t_eval: evaluation time points (eval_locs,)
        """
        def quad_integrate(a, b, sigma_pts, weights):
            # transform interval
            sigma_pts = 0.5*(sigma_pts + 1)*(b - a) + a
            weights *= jnp.prod( 0.5*(b - a) )
            return sigma_pts, weights

        vvinterp = vmap(vmap(jnp.interp, (None, None, 0), 0), (None, None, 2), 2)
        
        # compute rho(tau) for integral
        tau_pts = jnp.linspace(0.0, 1.0, int_eval_pts)
        t_pts, log_dt_dtau_pts = self.log_time_transform(tau_pts, inverse=True)

        log_rhotau_pts = self._transform_gp_sample(prng_state, tau_pts, x_cond, prior)
        log_rhot_pts = log_rhotau_pts - log_dt_dtau_pts[None, :, None]
        
        # compute rho(tau) and rho(t)
        rhotau_pts = jnp.exp(log_rhotau_pts)
        rhot_pts = jnp.exp(log_rhot_pts)
        
        # evaluation locs
        tau_eval, _ = self.log_time_transform(t_eval, inverse=False)
        rhot_eval = vvinterp(t_eval, t_pts, rhot_pts)

        # compute cumulative intensity int rho(tau) dtau
        sigma_pts, weights = gauss_legendre(1, num_quad_pts)
        locs, ws = vmap(quad_integrate, (None, 0, None, None), (0, 0))(
            0., tau_eval, sigma_pts, weights)

        quad_rhotau_eval = vmap(vvinterp, (1, None, None), 2)(locs[..., 0], tau_pts, rhotau_pts)
        int_rhotau_eval = (quad_rhotau_eval * ws[None, ..., None]).sum(-2)

        # normalizer
        locs, ws = quad_integrate(0., 1., sigma_pts, weights)
        quad_rho = vvinterp(locs[:, 0], tau_pts, rhotau_pts)
        int_rho = (quad_rho * ws[None, :, None]).sum(-2)
        normalizer = 1. - jnp.exp(-int_rho)  # (num_samps, out_dims)

        # compute renewal density
        exp_negintrhot = jnp.exp(-int_rhotau_eval)
        renewal_density = rhot_eval * exp_negintrhot / normalizer[:, None, :]
        return renewal_density
        
    
    
    
    
    
    
    
    
class FactorizedSwitchingSSGP(module):
    """
    The input-ouput (IO) mapping is deterministic or stochastic
    Switching prior
    
    Sample from prior, then use cubature for E_q(f|x) log p(y|f)
    
    Examples: GPFA, GLM (deterministic), GPLVM (stochastic)
    """

    def __init__(self, dtype=jnp.float32):
        super().__init__()
        self.dtype = dtype

        self.x_dims = state_space.kernel.out_dims
        self.state_space = state_space

        self.y = None  # no training data set
        
        
        
class FactorizedDTGPSSM(module):
    """
    GPSSM ELBO
    
    Sample from prior, then use cubature for E_q(f|x) log p(y|f)
    
    Examples: GPFA, GLM (deterministic), GPLVM (stochastic)
    """

    def __init__(self, dtype=jnp.float32):
        super().__init__()
        self.dtype = dtype

        self.x_dims = state_space.kernel.out_dims
        self.state_space = state_space

        self.y = None  # no training data set
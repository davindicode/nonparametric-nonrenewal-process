import numbers
import math

import numpy as np
#import scipy.special as scsps
#import scipy.stats as scstats

import jax
import jax.numpy as jnp
#import jax.scipy as jsc
from jax.scipy.special import erf, gammaln

from tqdm.autonotebook import tqdm

from .base import RenewalLikelihood

_log_twopi = math.log( 2 * math.pi )



### sampling ###
def gen_IRP(interval_dist, rate, dt, samples=100):
    """
    Sample event times from an Inhomogenous Renewal Process with a given rate function
    samples is an algorithm parameter, should be around the expect number of spikes
    Assumes piecewise constant rate function

    Samples intervals from :math:`q(\Delta)`, parallelizes sampling

    :param np.ndarray rate: (trials, neurons, timestep)
    :param ISI_dist interval_dist: renewal interval distribution :math:`q(\tau)`
    :returns: event times as integer indices of the rate array time dimension
    :rtype: list of spike time indices as indexed by rate time dimension
    """
    sim_samples = rate.shape[2]
    N = rate.shape[1]  # neurons
    trials = rate.shape[0]
    T = (
        np.transpose(np.cumsum(rate, axis=-1), (2, 0, 1)) * dt
    )  # actual time to rescaled, (time, trials, neuron)

    psT = 0
    sT_cont = []
    while True:

        sT = psT + np.cumsum(
            interval_dist.sample(
                (
                    samples,
                    trials,
                )
            ),
            axis=0,
        )
        sT_cont.append(sT)

        if not (T[-1, ...] >= sT[-1, ...]).any():  # all False
            break

        psT = np.tile(sT[-1:, ...], (samples, 1, 1))

    sT_cont = np.stack(sT_cont, axis=0).reshape(-1, trials, N)
    samples_tot = sT_cont.shape[0]
    st = []

    iterator = tqdm(range(samples_tot), leave=False)
    for ss in iterator:  # AR assignment
        comp = np.tile(sT_cont[ss : ss + 1, ...], (sim_samples, 1, 1))
        st.append(np.argmax((comp < T), axis=0))  # convert to rescaled time indices

    st = np.array(st)  # (samples_tot, trials, neurons)
    st_new = []
    for st_ in st.reshape(samples_tot, -1).T:
        if not st_.any():  # all zero
            st_new.append(np.array([]).astype(int))
        else:  # first zeros in this case counts as a spike indices
            for k in range(samples_tot):
                if st_[-1 - k] != 0:
                    break
            if k == 0:
                st_new.append(st_)
            else:
                st_new.append(st_[:-k])

    return st_new  # list of len trials x neurons



### renewal densities ###
class Gamma(RenewalLikelihood):
    """
    Gamma renewal process
    """
    shape: jnp.ndarray

    def __init__(
        self,
        neurons,
        dt,
        link_fn,
        shape,
        array_type=jnp.float32,
    ):
        """
        Renewal parameters shape can be shared for all neurons or independent.
        """
        super().__init__(neurons, dt, inv_link, array_type)
        self.shape = jnp.array(shape, dtype=self.array_type)

    def apply_constraints(self):
        """
        constrain shape parameter in numerically stable regime
        """
        def update(shape):
            return jnp.minimum(jnp.maximum(shape, 1e-5), 2.5)
        
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.shape,
            model,
            replace_fn=lambda _: kernel,
        )

        return model
    
    def log_renewal_density(self, ISI):
        """
        # d_Lambda_i = rates[:self.spiketimes[0]].sum()*self.dt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
        # d_Lambda_f = rates[self.spiketimes[ii]:].sum()*self.dt
        # l_start = jnp.empty((len(neuron)), device=self.dt.device)
        # l_end = jnp.empty((len(neuron)), device=self.dt.device)
        # l_start[n_enu] = jnp.log(sps.gammaincc(self.shape.item(), d_Lambda_i))
        # l_end[n_enu] = jnp.log(sps.gammaincc(self.shape.item(), d_Lambda_f))
        """
        shape_ = self.shape.expand(1, self.F_dims)[:, neuron]
        
        #shape = self.shape[n].data.cpu().numpy()
        intervals = jnp.zeros((samples_, len(neuron)))
        T = jnp.empty(
            (samples_, len(neuron)), 
        )  # MC samples, neurons
        l_Lambda = jnp.empty((samples_, len(neuron)))
        
        if len(isi) > 0:  # nonzero number of ISIs
            intervals[tr :: self.trials, n_enu] = isi.shape[-1]
            T[tr :: self.trials, n_enu] = isi.sum(-1)
            l_Lambda[tr :: self.trials, n_enu] = jnp.log(isi + 1e-12).sum(-1)

        else:
            T[
                tr :: self.trials, n_enu
            ] = 0  # TODO: minibatching ISIs approximate due to b.c.
            l_Lambda[tr :: self.trials, n_enu] = 0
                
        ll = (
            -(shape_ - 1) * l_Lambda
            + T
            + intervals[None, :] * gammaln(shape_)
        )
        
        return ll
    
    def cum_renewal_density(self, ISI):
        return
    
    def log_conditional_intensity(self, ISI):
        return

    def nll(self, rescaled_ISI, neuron):
        """
        Gamma case, approximates the spiketrain NLL (takes dt into account for NLL).
        
        Ignore the end points of the spike train
        

        :param np.ndarray neuron: fit over given neurons, must be an array
        :param jnp.ndarray F_mu: F_mu product with shape (samples, neurons, timesteps)
        :param jnp.ndarray F_var: variance of the F_mu values, same shape
        :param int b: batch number
        :param np.ndarray neuron: neuron indices that are used
        :param int samples: number of MC samples for likelihood evaluation
        :returns: NLL array over sample dimensions
        :rtype: jnp.ndarray
        """
        samples_ = n_l_rates.shape[
            0
        ]  # ll_samplesxcov_samples, in case of trials trial_num=cov_samples
        for tr, isis in enumerate(rISI):  # over trials
            for n_enu, isi in enumerate(isis):  # over neurons
                ll = self.log_renewal_density()
                
        nll = - n_l_rates - ll
        return nll.sum(1, keepdims=True)  # sum over neurons, keep as dummy time index

    def objective(self, spiketimes, pre_rates, covariates, neuron, num_ISIs):
        """
        :param jnp.ndarray pre_rates: pre-link rates (mc, out_dims, ts)
        :param List spiketimes: list of spike time indices arrays per neuron
        :param jnp.ndarray covariates: covariates time series (mc, out_dims, ts, in_dims)
        """
        mc, ts = covariates.shape[0], covariates.shape[2]
        
        # map posterior samples
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
            
        # NLL
        ll = jnp.nansum(self.log_renewal_density(rISI), axis=2)  # (mc, out_dims)
        
        nll = - n_l_rates - ll
        return nll
    
    

class LogNormal(RenewalLikelihood):
    """
    Log-normal ISI distribution
    Ignores the end points of the spike train in each batch
    """

    def __init__(
        self,
        neurons,
        inv_link,
        sigma,
        array_type=jnp.float32,
    ):
        """
        :param np.ndarray sigma: :math:`$sigma$` parameter which is > 0
        """
        super().__init__(neurons, inv_link, array_type)
        self.sigma = jnp.array(sigma, dtype=self.array_type)

    def apply_constraints(self):
        """
        constrain sigma parameter in numerically stable regime
        """
        def update(sigma):
            return jnp.maximum(sigma, 1e-5)
        
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.sigma,
            model,
            replace_fn=lambda _: kernel,
        )

        return model
    
    def log_renewal_density(self, ISI):
        """
        # d_Lambda_i = rates[:self.spiketimes[0]].sum()*self.dt
        # d_Lambda_f = rates[self.spiketimes[ii]:].sum()*self.dt
        # l_start = jnp.empty((len(neuron)), device=self.dt.device)
        # l_end = jnp.empty((len(neuron)), device=self.dt.device)
        # l_start[n_enu] = jnp.log(sps.gammaincc(self.shape.item(), d_Lambda_i))
        # l_end[n_enu] = jnp.log(sps.gammaincc(self.shape.item(), d_Lambda_f))
        """
        sigma_ = self.sigma.expand(1, self.F_dims)[:, neuron]
        
        l_Lambda = jnp.empty((samples_, len(neuron)))
        quad_term = jnp.empty((samples_, len(neuron)))
        norm_term = jnp.empty((samples_, len(neuron)))
        for tr, isis in enumerate(rISI):
            for n_enu, isi in enumerate(isis):
                if len(isi) > 0:  # nonzero number of ISIs
                    intervals = isi.shape[1]
                    l_Lambda[tr :: self.trials, n_enu] = jnp.log(isi + 1e-12).sum(-1)
                    quad_term[tr :: self.trials, n_enu] = 0.5 * (
                        (jnp.log(isi + 1e-12) / sigma_[:, n_enu : n_enu + 1]) ** 2
                    ).sum(
                        -1
                    )  # -mu_[:, n_enu:n_enu+1]
                    norm_term[tr :: self.trials, n_enu] = intervals * (
                        jnp.log(sigma_[0, n_enu]) + 0.5 * _log_twopi
                    )

                else:
                    l_Lambda[tr :: self.trials, n_enu] = 0
                    quad_term[tr :: self.trials, n_enu] = 0
                    norm_term[tr :: self.trials, n_enu] = 0

        ll = norm_term + l_Lambda + quad_term
        return ll
    
    def cum_renewal_density(self, ISI):
        return
    
    def log_conditional_intensity(self, ISI):
        return

    def set_Y(self, spikes, batch_info):
        super().set_Y(spikes, batch_info)

    def nll(self, n_l_rates, rISI, neuron):
        """
        Computes the log Normal distribution

        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).

        :param jnp.ndarray n_l_rates: log rates at spike times (samples, neurons, timesteps)
        :param jnp.ndarray rISI: modified rate rescaled ISIs
        :param np.ndarray neuron: neuron indices that are used
        :returns: spike train negative log likelihood of shape (timesteps, samples (dummy dimension))
        :rtype: jnp.ndarray
        """
        samples_ = n_l_rates.shape[0]
                    
        nll = -n_l_rates + ll
        return nll.sum(1, keepdims=True)

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        return super().objective(
            F_mu,
            F_var,
            XZ,
            b,
            neuron,
            jnp.exp(-self.sigma**2 / 2.0),
            samples=samples,
            mode=mode,
        )

    def ISI_dist(self, n):
        sigma = self.sigma[n].data.cpu().numpy()
        return ISI_logNormal(sigma, scale=np.exp(sigma**2 / 2.0))


    
    
class InverseGaussian(RenewalLikelihood):
    """
    Inverse Gaussian ISI distribution
    Ignores the end points of the spike train in each batch
    """
    mu: jnp.ndarray

    def __init__(
        self,
        neurons,
        inv_link,
        mu,
        array_type=jnp.float32,
    ):
        """
        :param np.ndarray mu: :math:`$mu$` parameter which is > 0
        """
        super().__init__(neurons, inv_link, array_type)
        self.mu = jnp.array(mu, dtype=self.array_type)
    
    def apply_constraints(self):
        """
        constrain sigma parameter in numerically stable regime
        """
        def update(mu):
            return jnp.maximum(mu, 1e-5)
        
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.mu,
            model,
            replace_fn=lambda _: kernel,
        )

        return model
        
    def log_renewal_density(self, n):
        """
        Note the scale parameter here is the inverse of the scale parameter in nll(), as the scale
        parameter here is :math:`\tau/s` while in nll() is refers to :math:`d\tau = s*r(t) \, \mathrm{d}t`
        """
        mu_ = self.mu.expand(1, self.F_dims)[:, neuron]
        
        l_Lambda = jnp.empty((samples_, len(neuron)))
        quad_term = jnp.empty((samples_, len(neuron)))
        norm_term = jnp.empty((samples_, len(neuron)))
        for tr, isis in enumerate(rISI):
            for n_enu, isi in enumerate(isis):
                if len(isi) > 0:  # nonzero number of ISIs
                    intervals = isi.shape[1]
                    l_Lambda[tr :: self.trials, n_enu] = jnp.log(isi + 1e-12).sum(-1)
                    quad_term[tr :: self.trials, n_enu] = 0.5 * (
                        ((isi - mu_[:, n_enu : n_enu + 1]) / mu_[:, n_enu : n_enu + 1])
                        ** 2
                        / isi
                    ).sum(
                        -1
                    )  # (lambd_[:, n_enu:n_enu+1])
                    norm_term[tr :: self.trials, n_enu] = intervals * (
                        0.5 * _log_twopi
                    )  # - 0.5*jnp.log(lambd_[0, n_enu])

                else:
                    l_Lambda[tr :: self.trials, n_enu] = 0
                    quad_term[tr :: self.trials, n_enu] = 0
                    norm_term[tr :: self.trials, n_enu] = 0

        ll = norm_term + 1.5 * l_Lambda + quad_term
        return ll
    
    def cum_renewal_density(self, ISI):
        return
    
    def log_conditional_intensity(self, ISI):
        return

        
    def set_Y(self, spikes, batch_info):
        super().set_Y(spikes, batch_info)

    def nll(self, n_l_rates, rISI, neuron):
        """
        :param jnp.ndarray F_mu: F_mu product with shape (samples, neurons, timesteps)
        :param jnp.ndarray F_var: variance of the F_mu values, same shape
        :param int b: batch number
        :param np.ndarray neuron: neuron indices that are used
        :param int samples: number of MC samples for likelihood evaluation
        """
        
        samples_ = n_l_rates.shape[0]
        
        nll = -n_l_rates + ll 
        return nll.sum(1, keepdims=True)

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        return super().objective(
            F_mu, F_var, XZ, b, neuron, 1.0 / self.mu, samples=samples, mode=mode
        )
import math
from functools import partial


import jax.numpy as np

from jax.scipy.special import erf, gammaln

from ..utils.jax import expsum, mc_sample, sigmoid, softplus, softplus_inv
from ..utils.linalg import gauss_hermite, get_blocks, inv

from .base import FactorizedLikelihood
from .factorized import Gaussian, ZeroInflatedPoisson, NegativeBinomial, ConwayMaxwellPoisson

_log_twopi = math.log(2 * math.pi)



class HeteroscedasticGaussian(Gaussian):
    """
    Heteroscedastic Gaussian likelihood
        p(y|f1,f2) = N(y|f1,link(f2)^2)
    """

    def __init__(self, out_dims, link_type = "softplus", array_type = jnp.float32):
        """
        :param link: link function, either 'exp' or 'softplus' (note that the link is modified with an offset)
        """
        assert link_type in ['log', 'softplus']
        super().__init__(out_dims, 2 * out_dims)

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood
        
        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        mu, var = f[:, 0], np.maximum(self.link_type(f[:, 1]) ** 2, 1e-8)
        ll = -0.5 * ((_log_twopi + np.log(var)) + (y - mu) ** 2 / var)
        return ll
    
    def variational_expectation(
        self, prng_state, y, f_mean, f_cov, jitter, approx_int_method, num_approx_pts, 
    ):
        # overwrite the exact homoscedastic case
        return FactorizedLikelihood.variational_expectation(
            self, prng_state, y, f_mean, f_cov, jitter, approx_int_method, num_approx_pts)
    
    
    
    

class HeteroscedasticZeroInflatedPoisson(ZeroInflatedPoisson):
    """
    Heteroscedastic ZIP
    """

    def __init__(
        self,
        tbin,
        neurons,
        link_type,
        dispersion_mapping,
        array_type=jnp.float32, 
    ):
        super().__init__(tbin, neurons, link_type, None, array_type, strict_likelihood)

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood
        
        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        mu, var = f[:, 0], np.maximum(self.link_type(f[:, 1]) ** 2, 1e-8)
        ll = -0.5 * ((_log_twopi + np.log(var)) + (y - mu) ** 2 / var)
        return ll

    
    


class HeteroscedasticNegativeBinomial(NegativeBinomial):
    """
    Heteroscedastic NB
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        dispersion_mapping,
        array_type=jnp.float32, 
        strict_likelihood=True,
    ):
        super().__init__(tbin, neurons, inv_link, None, tensor_type, strict_likelihood)

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood
        
        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        mu, var = f[:, 0], jnp.maximum(jax.nn.softplus(f[:, 1]) ** 2, 1e-8)
        ll = -0.5 * ((_log_twopi + np.log(var)) + (y - mu) ** 2 / var)
        return ll


class HeteroscedasticConwayMaxwellPoisson(ConwayMaxwellPoisson):
    """
    Heteroscedastic CMP
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        dispersion_mapping,
        array_type=jnp.float32, 
        J=100,
        strict_likelihood=True,
    ):
        super().__init__(
            tbin, neurons, inv_link, None, tensor_type, J, strict_likelihood
        )

    def log_likelihood(self, f, y):
        """
        Evaluate the log-likelihood
        
        :param jnp.ndarray f: input values (obs_dims, num_f_per_dims)
        :return:
            log likelihood of shape (approx_points,)
        """
        mu, var = f[:, 0], jnp.maximum(jax.nn.softplus(f[:, 1]) ** 2, 1e-8)
        ll = -0.5 * ((_log_twopi + np.log(var)) + (y - mu) ** 2 / var)
        return ll

    
    
class UniversalCount(CountLikelihood):
    """
    Universal count distribution with finite cutoff at max_count
    """
    K: int
    C: int
        
    # final layer mapping
    W: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, out_dims, C, K, tbin, array_type = jnp.float32):
        """
        :param int K: max spike count
        """
        super().__init__(out_dims, C * out_dims, tbin, array_type)
        self.K = K
        self.C = C
        
        self.W = W  # maps from NxC to NxK
        self.b = b
    
    def check_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for fast log-likelihood evaluation. 
        Batched spikes will be a list of tensors of shape (trials, neurons, time) with trials 
        set to 1 if input has no trial dimension (e.g. continuous recording).
        
        :param np.ndarray spikes: becomes a list of [neuron_dim, batch_dim]
        :param int/list batch_size: 
        :param int filter_len: history length of the GLM couplings (1 indicates no history coupling)
        """
        if self.K < spikes.max():
            raise ValueError('Maximum count is exceeded in the spike count data')
        super().set_Y(spikes, batch_info)
                
        
    def onehot_to_counts(self, onehot):
        """
        Convert one-hot vector representation of counts. Assumes the event dimension is the last.
        
        :param jnp.ndarray onehot: one-hot vector representation of shape (..., event)
        :returns: spike counts
        :rtype: jnp.ndarray
        """
        counts = jnp.zeros(*onehot.shape[:-1], device=onehot.device)
        inds = jnp.where(onehot)
        counts[inds[:-1]] = inds[-1].float()
        return counts

    
    def counts_to_onehot(self, counts):
        """
        Convert counts to one-hot vector representation. Adds the event dimension at the end.
        
        :param jnp.ndarray counts: spike counts of some tensor shape
        :param int max_counts: size of the event dimension (max_counts + 1)
        :returns: one-hot representation of shape (counts.shape, event)
        :rtype: jnp.ndarray
        """
        km = self.K+1
        onehot = jnp.zeros(*counts.shape, km, device=counts.device)
        onehot_ = onehot.view(-1, km)
        g = onehot_.shape[0]
        onehot_[np.arange(g), counts.flatten()[np.arange(g)].long()] = 1
        return onehot_.view(*onehot.shape)
    
    
    def get_logp(self, F_mu, neuron):
        """
        Compute count probabilities from the rate model output.
        
        :param jnp.ndarray F_mu: the F_mu product output of the rate model (samples and/or trials, F_dims, time)
        :returns: log probability tensor
        :rtype: tensor of shape (samples and/or trials, n, t, c)
        """
        T = F_mu.shape[-1]
        samples = F_mu.shape[0]
        a = self.mapping_net(F_mu.permute(0, 2, 1).reshape(samples*T, -1), neuron) # samplesxtime, NxK
        log_probs = self.lsoftm(a.view(samples, T, -1, self.K+1).permute(0, 2, 1, 3))
        return log_probs
    
    
    def log_likelihood(self, f, y):
        """
        Evaluate the softmax from 0 to K with linear mapping from f
        """
        logp_cnts = jax.nn.log_softmax(W @ f + b, axis=-1)
        
        tar = self.counts_to_onehot(spikes)
        nll = -(tar*logp).sum(-1)
        
        return logp_cnts[int(y)]
    
    
#     def _neuron_to_F(self, neuron):
#         """
#         Access subset of neurons in expanded space.
#         """
#         neuron = self._validate_neuron(neuron)
#         if len(neuron) == self.neurons:
#             F_dims = list(range(self.F_dims))
#         else: # access subset of neurons
#             F_dims = list(np.concatenate([np.arange(n*self.C, (n+1)*self.C) for n in neuron]))
            
#         return F_dims

    
    def sample_Y(self, F_mu, neuron, XZ=None):
        """
        Sample from the categorical distribution.
        
        :param numpy.array log_probs: log count probabilities (trials, neuron, timestep, counts), no 
                                      need to be normalized
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        F_dims = self._neuron_to_F(neuron)
        log_probs = self.get_logp(jnp.array(F_mu[:, F_dims, :], dtype=self.array_type))
        c_dist = mdl.distributions.Categorical(logits=log_probs)
        cnt_prob = jnp.exp(log_probs)
        return c_dist.sample().numpy()
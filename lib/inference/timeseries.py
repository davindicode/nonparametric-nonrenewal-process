from typing import List, Union

import numpy as np
import jax.numpy as jnp

from ..base import module
from ..GP.markovian import GaussianLTI
from ..GP.switching import SwitchingLTI
from ..GP.gpssm import DTGPSSM



class BatchedTimeSeries:
    """
    Data with loading functionality
    
    Allows for filtering
    """

    def __init__(self, timestamps, covariates, ISIs, observations, batch_size, filter_length=0):
        """
        :param np.ndarray timestamps: (ts,)
        :param np.ndarray covariates: (ts, x_dims)
        :param np.ndarray ISIs: (out_dims, ts, order)
        :param np.ndarray observations: (out_dims, ts)
        """
        pts = len(timestamps)
        
        # checks
        assert covariates.shape[0] == pts
        if ISIs is not None:
            assert ISIs.shape[1] == pts
        assert observations.shape[1] == pts + filter_length
        
        self.batches = int(np.ceil(pts / batch_size))
        self.batch_size = batch_size
        self.filter_length = filter_length

        self.timestamps = timestamps
        self.covariates = covariates
        self.ISIs = ISIs
        self.observations = observations
        
    def load_batch(self, batch_index):
        t_inds = slice(batch_index*self.batch_size, (batch_index + 1)*self.batch_size)
        y_inds = slice(batch_index*self.batch_size, (batch_index + 1)*self.batch_size + self.filter_length)
        
        ts = self.timestamps[t_inds]
        xs = self.covariates[t_inds]
        deltas = self.ISIs[:, t_inds]
        ys = self.observations[:, t_inds]
        return ts, xs, deltas, ys
    
    
    
class BatchedTrials:
    """
    Subsample over batches
    """
    
    def __init__(self, timestamps, covariates, ISIs, observations, batch_size, filter_length=0):
        """
        :param np.ndarray timestamps: (ts,)
        :param np.ndarray covariates: (ts, x_dims)
        :param np.ndarray ISIs: (out_dims, ts, order)
        :param np.ndarray observations: (out_dims, ts)
        """
        pts = len(timestamps)

    
    
    
class GaussianLatentObservedSeries(module):
    """
    Time series with latent GP and observed covariates
    """
    
    ssgp: Union[GaussianLTI, None]
    lat_dims: List[int]
    obs_dims: List[int]
    x_dims: int

    def __init__(self, ssgp, lat_dims, obs_dims, array_type=jnp.float32):
        if ssgp is not None:  # checks
            assert ssgp.array_type == array_type
            assert len(lat_dims) == ssgp.kernel.out_dims
            
        super().__init__(array_type)
        self.ssgp = ssgp
        self.lat_dims = lat_dims
        self.obs_dims = obs_dims
        self.x_dims = len(self.lat_dims) + len(self.obs_dims)

    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = super().apply_constraints()
        model = eqx.tree_at(
            lambda tree: tree.ssgp,
            model,
            replace_fn=lambda obj: obj.apply_constraints(),
        )

        return model
    
    def sample_prior(self, prng_state, num_samps, timestamps, x_eval, jitter):
        """
        Combines observed inputs with latent trajectories
        
        :param jnp.ndarray timestamps: time stamps of inputs (ts,)
        :param jnp.ndarray x_eval: inputs for evaluation (ts, x_dims)
        """
        ts = len(timestamps)
        if len(self.obs_dims) > 0:
            x_eval = jnp.broadcast_to(x_eval, (num_samps, ts, len(self.obs_dims)))
            
        if len(self.lat_dims) == 0:
            x = x_eval
            
        else:
            x_samples = self.ssgp.sample_prior(prng_state, num_samps, timestamps, jitter)[..., 0]
            
            if len(self.obs_dims) == 0:
                x = x_samples
                
            else:
                x = jnp.empty((num_samps, ts,  self.x_dims))
                x = x.at[..., self.obs_dims].set(x_eval)
                x = x.at[..., self.lat_dims].set(x_samples)
            
        return x  # (num_samps, ts, x_dims)

    def sample_posterior(self, prng_state, num_samps, timestamps, x_eval, jitter, compute_KL):
        """
        Combines observed inputs with latent trajectories
        
        :param jnp.ndarray timestamps: 
        """
        ts = len(timestamps)
        if len(self.obs_dims) > 0:
            x_eval = jnp.broadcast_to(x_eval, (num_samps, ts, len(self.obs_dims)))
            
        if len(self.lat_dims) == 0:
            x, KL = x_eval, 0.
            
        else:
            x_samples, KL = self.ssgp.sample_posterior(
                prng_state, num_samps, timestamps, jitter, compute_KL)  # (tr, time, x_dims)
            x_samples = x_samples[..., 0]
            
            if len(self.obs_dims) == 0:
                x = x_samples
            
            else:
                x = jnp.empty((num_samps, len(timestamps),  self.x_dims))
                x = x.at[..., self.obs_dims].set(x_eval)
                x = x.at[..., self.lat_dims].set(x_samples)
            
        return x, KL

    def marginal_posterior(self, num_samps, timestamps, x_eval, jitter, compute_KL):
        """
        Combines observed inputs with latent marginal samples
        """
        ts = len(timestamps)
        if len(self.obs_dims) > 0:
            x_eval = jnp.broadcast_to(x_eval, (num_samps, ts, len(self.obs_dims)))
        
        if len(self.lat_dims) == 0:
            x_mean, KL = x_eval, 0.
            x_cov = jnp.zeros_like(x)
            
        else:
            post_mean, post_cov, KL = self.ssgp.evaluate_posterior(
                timestamps, False, compute_KL, jitter)
            post_mean, post_cov = post_mean[..., 0], post_cov[..., 0]  # (tr, time, x_dims)
            
            if len(self.obs_dims) == 0:
                x_mean, x_cov = post_mean, post_cov
            
            else:
                x_mean, x_cov, KL = jnp.empty((num_samps, len(timestamps))), jnp.empty((num_samps, len(timestamps))), 0.
                x_mean = x_mean.at[..., self.obs_dims].set(x_eval)
                x_mean = x_mean.at[..., self.lat_dims].set(post_mean)
                x_cov = x_cov.at[..., self.obs_dims].set(jnp.zeros_like(x_eval))
                x_cov = x_cov.at[..., self.lat_dims].set(post_cov)
                
        return x_mean, x_cov, KL
    
    
    def sample_marginal_posterior(self, prng_state, num_samps, timestamps, x_eval, jitter, compute_KL):
        x_mean, x_cov, KL = self.marginal_posterior(
            prng_state, num_samps, timestamps, x_eval, jitter, compute_KL)  # (num_samps, obs_dims, ts, 1)
        
        # conditionally independent sampling
        x_std = safe_sqrt(x_cov)
        x = x_mean + jr.normal(prng_state, shape=x_mean.shape) * x_std
        return x, KL
        
        
#     switchgp: Union[SwitchingLTI, None]
#     gpssm: Union[DTGPSSM, None]
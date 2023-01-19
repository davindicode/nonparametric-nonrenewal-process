from typing import List, Union

import numpy as np

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ..base import module
from ..GP.markovian import GaussianLTI
from ..GP.switching import SwitchingLTI
from ..GP.gpssm import DTGPSSM
from ..utils.jax import safe_sqrt



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
        :param np.ndarray observations: includes filter history (out_dims, ts + filter_length)
        """
        pts = len(timestamps)
        
        # checks
        if covariates is not None:
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
        y_inds = slice(batch_index*self.batch_size + self.filter_length, 
                       (batch_index + 1)*self.batch_size + self.filter_length)
        
        ts = self.timestamps[t_inds]
        xs = self.covariates[t_inds] if self.covariates is not None else None
        deltas = self.ISIs[:, t_inds] if self.ISIs is not None else None
        
        ys = self.observations[:, y_inds]
        if self.filter_length > 0:
            filt_inds = slice(batch_index*self.batch_size, 
                              batch_index*self.batch_size + self.filter_length + ys.shape[1] - 1)
            ys_filt = self.observations[:, filt_inds]  # leave out last time step (causality)
        else:
            ys_filt = None
            
        return ts, xs, deltas, ys, ys_filt
    
    
    
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
    diagonal_cov: bool

    def __init__(self, ssgp, lat_dims, obs_dims, diagonal_cov=False, array_type=jnp.float32):
        if ssgp is not None:  # checks
            assert ssgp.array_type == array_type
            assert len(lat_dims) == ssgp.kernel.out_dims
            
        super().__init__(array_type)
        self.ssgp = ssgp
        self.lat_dims = lat_dims
        self.obs_dims = obs_dims
        self.x_dims = len(self.lat_dims) + len(self.obs_dims)
        self.diagonal_cov = diagonal_cov

    def apply_constraints(self):
        """
        Constrain parameters in optimization
        """
        model = super().apply_constraints()
        if model.ssgp is not None:
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
        
        :param jnp.ndarray x_eval: observed covariates (ts, obs_dims)
        :return:
            x_mean (tr, time, x_dims)
            x_cov (tr, time, x_dims, 1 or x_dims) depending on diagonal_cov
            KL divergence (scalar)
        """
        ts = len(timestamps)
        if len(self.obs_dims) > 0:
            x_eval = jnp.broadcast_to(x_eval, (num_samps, ts, len(self.obs_dims)))
        
        if len(self.lat_dims) == 0:
            x_mean, KL = x_eval, 0.
            x_cov = jnp.zeros_like(x_eval)[..., None]
            
        else:
            post_mean, post_cov, KL = self.ssgp.evaluate_posterior(
                timestamps, False, compute_KL, jitter)
            post_mean = post_mean[..., 0]  # (tr, time, x_dims)
            
            if len(self.obs_dims) == 0:
                x_mean, x_cov = post_mean, post_cov
            
            else:
                x_mean, KL = jnp.empty((num_samps, len(timestamps), self.x_dims), dtype=self.array_dtype()), 0.
                x_mean = x_mean.at[..., self.obs_dims].set(x_eval)
                x_mean = x_mean.at[..., self.lat_dims].set(post_mean)
                
                if self.diagonal_cov:
                    x_cov = jnp.empty((num_samps, len(timestamps), self.x_dims, 1), dtype=self.array_dtype())
                    x_cov = x_cov.at[..., self.obs_dims, 0].set(jnp.zeros_like(x_eval))
                    x_cov = x_cov.at[..., self.lat_dims, 0].set(vmap(vmap(jnp.diag))(post_cov))
                    
                else:
                    x_cov = jnp.zeros((num_samps, len(timestamps), self.x_dims, self.x_dims), dtype=self.array_dtype())
                    acc = jnp.array(self.lat_dims)[:, None].repeat(len(self.lat_dims), axis=1)
                    x_cov = x_cov.at[..., acc, acc.T].set(post_cov)
                
        return x_mean, x_cov, KL
    
    
    def sample_marginal_posterior(self, prng_state, num_samps, timestamps, x_eval, jitter, compute_KL):
        x_mean, x_cov, KL = self.marginal_posterior(
            num_samps, timestamps, x_eval, jitter, compute_KL)
        
        # conditionally independent sampling
        if self.diagonal_cov:
            x_std = safe_sqrt(x_cov)
            x = x_mean + x_std[..., 0] * jr.normal(prng_state, shape=x_mean.shape)  # (num_samps, ts, x_dims)
        else:
            eps = jitter * jnp.eye(self.x_dims)[None, None]
            Lcov = cholesky(x_cov + eps)
            x = x_mean + (Lcov @ jr.normal(prng_state, shape=x_mean.shape))[..., 0]  # (num_samps, ts, x_dims)
        return x, KL
        
        
#     switchgp: Union[SwitchingLTI, None]
#     gpssm: Union[DTGPSSM, None]
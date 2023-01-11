from typing import List, Union
import jax.numpy as jnp

from ..base import module
from ..GP.markovian import GaussianLTI
from ..GP.switching import SwitchingLTI
from ..GP.gpssm import DTGPSSM



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
            assert len(lat_dims) == ssgp.f_dims
            
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
        """
        x = jnp.empty((self.x_dims))
        
        if len(self.obs_dims) > 0:
            x.append(x_eval)
            
        if len(self.lat_dims) > 0:  # self.ssgp is not None
            x.append(self.ssgp.sample_prior(prng_state, num_samps, t_eval, jitter))
        
        return x

    def sample_posterior(self, prng_state, num_samps, x_eval, t_eval, jitter, compute_KL):
        """
        Combines observed inputs with latent trajectories
        """
        x, KL = [], 0.
        
        if x_eval is not None:
            x.append(x_eval)
            
        if self.ssgp is not None:
            x_samples, KL_x = self.ssgp.sample_posterior(
                prng_state, num_samps, t_eval, jitter, compute_KL)  # (tr, time, N, 1)

            x.append(x_samples)
            KL += KL_x
            
        if len(x) > 0:
            x = jnp.concatenate(x, axis=-1)
        else:
            x = None
        return x, KL

    def marginal_posterior(self, prng_state, num_samps, x_eval, t_eval, jitter, compute_KL):
        """
        Combines observed inputs with latent marginal samples
        """
        x, x_cov, KL = [], [], 0.
        
        if x_eval is not None:
            x.append(x_eval)
            x_cov.append(jnp.zeros_like(x_eval))
            
        if self.ssgp is not None:  # filtering-smoothing
            post_mean, post_cov, KL_x = self.ssgp.evaluate_posterior(
                t_eval, False, compute_KL, jitter)
            post_mean = post_mean[..., 0]  # (time, tr, x_dims, 1)

            x.append(post_mean)
            x_cov.append(post_cov)
            KL += KL_x
        
        if len(x) > 0:
            x, x_cov = jnp.concatenate(x, axis=-1), jnp.concatenate(x_cov, axis=-1)
        else:
            x, x_cov = None, None
        return x, x_cov, KL
    
    
    def sample_marginal_posterior(self):
        x_mean, x_cov, KL_x = self._posterior_input_marginals(
            prng_state, num_samps, t_eval, jitter, compute_KL)  # (num_samps, obs_dims, ts, 1)
        
        # conditionally independent sampling
        x_std = safe_sqrt(x_cov)
        x_samples = x_mean + jr.normal(prng_state, shape=x_mean.shape) * x_std

        
        
#     switchgp: Union[SwitchingLTI, None]
#     gpssm: Union[DTGPSSM, None]
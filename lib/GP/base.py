import math
from functools import partial

from jax import lax, vmap
import jax.numpy as jnp
import jax.random as jr
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular

from ..base import module
from ..utils.jax import sample_gaussian_noise

from .kernels import Kernel, MarkovianKernel
from .linalg import mvn_conditional




class GP(module):
    """
    GP with function and RFF kernel
    """
    kernel: Kernel
    RFF_num_feats: int
    
    mean: jnp.ndarray
        
    def __init__(self, kernel, mean, RFF_num_feats):
        super().__init__()
        self.kernel = kernel
        self.mean = mean  # (out_dims,)
        self.RFF_num_feats = RFF_num_feats  # use random Fourier features
        
    def apply_constraints(self):
        """
        PSD constraint
        """
        kernel = self.kernel.apply_constraints(self.kernel)
        
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.kernel,
            model,
            replace_fn=lambda _: kernel,
        )

        return model
        
    
    def evaluate_conditional(self, x, x_obs, f_obs, mean_only, diag_cov, jitter):
        """
        Compute the conditional distribution
        
        :param jnp.array x: shape (time, num_samps, in_dims)
        :param jnp.array x_obs: shape (out_dims, obs_pts, in_dims)
        :return:
            conditional mean of shape (out_dims, num_samps, ts, 1)
            conditional covariance of shape (out_dims, num_samps, ts, 1)
        """
        cond_out = vmap(
            mvn_conditional, 
            (2, None, None, None, None, None, None), 
            1 if mean_only else (1, 1),
        )(
            x[None, ...], x_obs, f_obs, self.kernel.K, mean_only, diag_cov, jitter
        )
        
        if mean_only:
            return cond_out + self.mean[:, None, None, None]
        else:
            return cond_out[0] + self.mean[:, None, None, None], cond_out[1]
        
        
    def sample_prior(self, prng_state, x, jitter):
        """
        Prior distribution p(f(x)) = N(0, K_xx)
        Can use approx_points as number of points

        :param jnp.array x: shape (time, num_samps, out_dims, in_dims)
        :return:
            sample of shape (time, num_samps, out_dims)
        """
        in_dims = self.kernel.in_dims
        out_dims = self.kernel.out_dims
        
        if x.ndim == 3:
            x = x[..., None, :]  # (time, num_samps, out_dims, in_dims)
        ts, num_samps = x.shape[:2]

        if self.RFF_num_feats > 0:  # random Fourier features
            prng_keys = jr.split(prng_state, 2)
            ks, amplitude = self.kernel.sample_spectrum(
                prng_keys[0], num_samps, self.RFF_num_feats
            )  # (num_samps, out_dims, feats, in_dims)
            phi = (
                2
                * jnp.pi
                * jr.uniform(
                    prng_keys[1], shape=(num_samps, out_dims, self.RFF_num_feats)
                )
            )

            samples = self.mean[None, None, :] + amplitude * jnp.sqrt(
                2.0 / self.RFF_num_feats
            ) * (jnp.cos((ks[None, ...] * x[..., None, :]).sum(-1) + phi)).sum(
                -1
            )  # (time, num_samps, out_dims)

        else:
            Kxx = vmap(self.kernel.K, (1, None, None), 1)(
                x.transpose(2, 1, 0, 3), None, False
            )  # (out_dims, num_samps, time, time)
            eps_I = jnp.broadcast_to(jitter * jnp.eye(ts), (out_dims, 1, ts, ts))
            cov = Kxx + eps_I
            mean = jnp.broadcast_to(
                self.mean[:, None, None, None], (out_dims, num_samps, ts, 1)
            )
            samples = sample_gaussian_noise(prng_state, mean, cov)[..., 0].transpose(
                2, 1, 0
            )

        return samples
    
    
    
class SSM(module):
    """
    Gaussian Linear Time-Invariant System
    
    Temporal multi-output kernels have separate latent processes that can be coupled.
    Spatiotemporal kernel modifies the process noise across latent processes, but dynamics uncoupled.
    Multi-output GPs generally mix latent processes via dynamics as well.
    """
    
    site_locs: jnp.ndarray  
    site_obs: jnp.ndarray
    site_Lcov: jnp.ndarray

    def __init__(self, site_locs, site_obs, site_Lcov):
        """
        :param module markov_kernel: (hyper)parameters of the state space model
        :param jnp.ndarray site_locs: means of shape (time, 1)
        :param jnp.ndarray site_obs: means of shape (time, x_dims, 1)
        :param jnp.ndarray site_Lcov: covariances of shape (time, x_dims, x_dims)
        """
        super().__init__()
        self.site_locs = site_locs
        self.site_obs = site_obs
        self.site_Lcov = site_Lcov
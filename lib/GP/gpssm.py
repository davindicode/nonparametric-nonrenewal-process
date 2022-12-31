import jax
from jax import lax, vmap
import jax.random as jr
import jax.numpy as jnp


from ..base import module
from .base import GP
from .markovian import LGSSM

    

# GPSSM
class DTGPSSM(module):
    """
    Discrete-time Gaussian process state-space model
    """
    
    dynamics_function: GP
    chol_process_noise: jnp.ndarray  # (out_dims, state_dims, state_dims)
        
    p0_mean: jnp.ndarray  # (state_dims, 1)
    p0_Lcov: jnp.ndarray  # (state_dims, state_dims)
        
    state_posterior: LGSSM
    
    def __init__(self, dynamics_function, chol_process_noise, p0_mean, p0_Lcov, state_posterior, array_type=jnp.float32):
        assert dynamics_function.kernel.out_dims == dynamics_function.kernel.in_dims
        super().__init__(array_type)
        self.dynamics_function = dynamics_function
        self.chol_process_noise = chol_process_noise
        
        self.p0_mean = p0_mean
        self.p0_Lcov = p0_Lcov
        
        self.state_posterior = state_posterior
#         if state_estimation == 'UKF':
#             self.evaluate_posterior = self.evaluate_posterior_UKF
#         elif state_estimation == 'vLDS':
#             self.evaluate_posterior = self.evaluate_posterior_vLDS
    
    def apply_constraints(self):
        """
        PSD constraint
        """
        GP = self.GP.apply_constraints(self.GP)
        
        model = jax.tree_map(lambda p: p, self)  # copy
        model = eqx.tree_at(
            lambda tree: tree.GP,
            model,
            replace_fn=lambda _: GP,
        )
        
        def update(Lcov):
            epdfunc = lambda x: enforce_positive_diagonal(x, lower_lim=1e-2)
            Lcov = vmap(epdfunc)(jnp.tril(Lcov))
            Lcov = jnp.tril(Lcov)
            return Lcov

        model = eqx.tree_at(
            lambda tree: tree.chol_process_noise,
            model,
            replace_fn=update,
        )
        
        model = eqx.tree_at(
            lambda tree: tree.p0_Lcov,
            model,
            replace_fn=update,
        )

        return model
       
    def sample_prior(self, prng_state, num_samps, timesteps, jitter):
        """
        Sample from the model prior
        
        :param jnp.ndarray x0: the initial state (num_samps, state_dims)
        :param t: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            x_samples: the prior samples (num_samps, ts, x_dims)
        """
        x_dims = self.dynamics_function.kernel.in_dims
        
        eps_I = jitter * jnp.eye(x_dims)

        prng_states = jr.split(prng_state, 1 + timesteps)  # (num_samps, 2)
        prng_state, procnoise_keys = prng_states[0], prng_states[1:]

        if self.dynamics_function.RFF_num_feats > 0:  # RFF prior sampling
            def step(carry, inputs):
                x, prng_prior = carry  # (num_samps, x_dims)
                prng_state = inputs

                fx = self.dynamics_function.sample_prior(
                    prng_prior, x[:, None, None, :], jitter)  # (samp, x_dims, 1)
                noise = self.chol_process_noise[None, ...] @ jr.normal(
                    prng_state, shape=(num_samps, x_dims, 1))
                x = x + fx[..., 0] + noise[..., 0]

                return (x, prng_prior), x

            x0 = (self.p0_Lcov[None, ...] @ jr.normal(
                prng_state, shape=(num_samps, x_dims, 1)
            ))[..., 0]  # (num_samps, x_dims)
            prng_state, _ = jr.split(prng_state)
            _, x_samples = lax.scan(step, init=(x0, prng_state), xs=procnoise_keys)
            
        else:  # autoregressive sampling using conditionals
            def step(carry, inputs):
                x, x_obs, f_obs = carry  # (1, num_samps, x_dims)
                prng_state = inputs
    
                x = x[None, ..., 0]

                qf_m, qf_v = self.dynamics_function.evaluate_conditional(
                    x, x_obs, f_obs, mean_only=False, diag_cov=True, jitter=1e-6)

                fx = qf_m + qf_v @ jr.normal(prng_state, shape=(num_samps, x_dims, 1))  # (out_dims, num_samps, 1)
                prng_state, _ = jr.split(prng_state)
                
                x_obs = jnp.concatenate((x_obs, x), axis=1)  # (out_dims, obs_pts, 1)
                f_obs = jnp.concatenate((f_obs, fx), axis=1)  # (out_dims, obs_pts, 1)
                
                noise = self.chol_process_noise[None, ...] @ jr.normal(prng_state, shape=(num_samps, x_dims, 1))
                x = x + fx + noise
                
                return (x, x_obs, f_obs), x

            x0 = self.p0_Lcov[None, ...] @ jr.normal(
                prng_state, shape=(num_samps, x_dims, 1)
            )
            carry = (x0, jnp.empty((x_dims, 0, x_dims)), jnp.empty((x_dims, 0, x_dims)))
            x_samples = jnp.empty((timesteps, num_samps, x_dims, 1))
            for t in range(timesteps):
                carry, x_sample = step(carry, procnoise_keys[t, ...])
                x_samples = x_samples.at[t, ...].set(x_sample)
            #_, x_samples = lax.scan(step, init=x0, xs=procnoise_keys)

        return x_samples.transpose(1, 0, 2)  # (num_samps, time, state_dims)
    
    
    def evaluate_posterior(self, ):
        """
        The augmented KL divergence includes terms due to the state-space mapping
        """
        return post_mean, post_cov, aug_KL
    
    
    def sample_posterior(self, prng_state, num_samps, timesteps, jitter):
        """
        """
        x_dims = self.dynamics_function.kernel.in_dims
        
        eps_I = jitter * jnp.eye(x_dims)

        prng_states = jr.split(prng_state, 1 + timesteps)  # (num_samps, 2)
        prng_state, procnoise_keys = prng_states[0], prng_states[1:]
        
        if self.dynamics_function.RFF_num_feats > 0:  # RFF prior sampling
            def step(carry, inputs):
                x, prng_prior = carry  # (num_samps, x_dims)
                prng_state = inputs

                fx, _ = self.dynamics_function.sample_posterior(
                    prng_prior, x[:, None, None, :], jitter, compute_KL=False)  # (samp, x_dims, 1)
                noise = self.chol_process_noise[None, ...] @ jr.normal(
                    prng_state, shape=(num_samps, x_dims, 1))
                x = x + fx[..., 0] + noise[..., 0]

                return (x, prng_prior), x

            x0 = (self.p0_Lcov[None, ...] @ jr.normal(
                prng_state, shape=(num_samps, x_dims, 1)
            ))[..., 0]  # (num_samps, x_dims)
            prng_state, _ = jr.split(prng_state)
            _, x_samples = lax.scan(step, init=(x0, prng_state), xs=procnoise_keys)
            
        else:  # autoregressive sampling using conditionals
            return
        
        return x_samples.transpose(1, 0, 2)  # (num_samps, time, state_dims)
        
        
#     def compute_jac(self, probe_state, probe_input):
#         """
#         :param jnp.ndarray probe_state: state of shape (evals, hidden_size)
#         """
#         f = lambda v, I: self.dv(1.0, v[None, :], I[None, :])[0, :]
#         J = jax.vmap(jax.jacfwd(f), 0, 0)(probe_state, probe_input)
#         return J


#     def find_slow_points(
#         model,
#         optim,
#         initial_states,
#         probe_inputs,
#         iters,
#     ):
#         probe_states = initial_states
#         opt_state = optim.init(probe_states)
#         loss_tracker = []

#         @partial(jit, static_argnums=())
#         def velocity_squared(model, probe_state, probe_input):
#             vel = model.dv(1.0, probe_state, probe_input)
#             return (vel**2).mean(0).sum()

#         iterator = tqdm(range(iters))
#         for ep in iterator:

#             loss, grads = jax.value_and_grad(velocity_squared, argnums=1)(
#                 model, probe_states, probe_inputs
#             )
#             loss = loss.item()
#             loss_tracker.append(loss)

#             updates, opt_state = optim.update(grads, opt_state)
#             probe_states = probe_states.at[...].add(updates)

#             loss_dict = {"loss": loss}
#             iterator.set_postfix(**loss_dict)

#         vel_squared = model.dv(1.0, probe_states, probe_inputs) ** 2
#         return probe_states, vel_squared, loss_tracker


#     def filter_slow_points(slow_points, tol=1e-5):
#         unique_slow_points = jnp.empty((0, slow_points.shape[1]))
#         inds_list = []
#         for en, s in enumerate(slow_points):
#             if len(unique_slow_points) > 0:
#                 dist_squared = ((s[None, :] - unique_slow_points) ** 2).sum(1)
#                 if jnp.min(dist_squared, axis=0) < tol:
#                     continue
#             unique_slow_points = jnp.append(unique_slow_points, s[None, :], axis=0)
#             inds_list.append(en)

#         return unique_slow_points, inds_list  # (num, dims)
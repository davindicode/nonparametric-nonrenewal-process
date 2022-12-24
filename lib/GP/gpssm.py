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
    
    def __init__(self, dynamics_function, chol_process_noise, p0_mean, p0_Lcov, state_posterior):
        super().__init__()
        self.dynamics_function = dynamics_function
        self.chol_process_noise = chol_process_noise
        
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

        return model
       
    def sample_prior(self, prng_state, x0, timesteps, jitter):
        """
        Sample from the model prior
        
        :param jnp.ndarray x0: the initial state (num_samps, state_dims)
        :param t: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            f_sample: the prior samples [S, N_samp]
        """
        num_samps = x0.shape[0]
        
        eps_I = jitter * jnp.eye(self.markov_kernel.state_dims)
        tsteps = timedata[0].shape[0]

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        if self.dynamics_function.self.RFF_num_feats > 0:  # RFF prior sampling
            def step(carry, inputs):
                x, prng_state = carry
                A, Q, prng_state = inputs

                num_samps = 10
                xx = np.linspace(-8., 8., 100)[:, None, None].repeat(num_samps, axis=1)

                obs_pts = 4
                x_obs = np.linspace(-5., 5., obs_pts)[None, :, None]
                f_obs = np.linspace(-5., 5., obs_pts)[None, :, None]

                pf_x = obs.sample_prior(prng_state, xx, jitter)  # (evals, samp, f_dim)


                q_samp = L @ jr.normal(prng_state, shape=(self.markov_kernel.state_dims, 1))
                m = A @ m + q_samp
                f = H @ m
                return m, f

            def sample_i(prng_state):
                m0 = cholesky(Pinf) @ jr.normal(
                    prng_state, shape=(self.markov_kernel.state_dims, 1)
                )
                _, f_sample = lax.scan(step, init=(x0, prng_state), xs=())
                return f_sample
            
        else:  # autoregressive sampling using conditionals
            def step(carry, inputs):
                m = carry
                A, Q, prng_state = inputs

                num_samps = 10
                xx = np.linspace(-8., 8., 100)[:, None, None].repeat(num_samps, axis=1)

                obs_pts = 4
                x_obs = np.linspace(-5., 5., obs_pts)[None, :, None]
                f_obs = np.linspace(-5., 5., obs_pts)[None, :, None]


                qf_m, qf_c = dynamics_function.evaluate_conditional(
                    xx, x_obs, f_obs, mean_only=False, diag_cov=False, jitter=1e-6)


                q_samp = L @ jr.normal(prng_state, shape=(self.markov_kernel.state_dims, 1))
                m = A @ m + q_samp
                f = H @ m
                return m, f

            def sample_i(prng_state):
                m0 = cholesky(Pinf) @ jr.normal(
                    prng_state, shape=(self.markov_kernel.state_dims, 1)
                )
                procnoise_keys = jr.split(prng_state, tsteps)
                _, f_sample = lax.scan(step, init=m0, xs=(As[:-1], Qs[:-1], procnoise_keys))
                return f_sample

        x_samples = vmap(sample_i, 0, 1)(prng_states)
        return x_samples  # (time, tr, state_dims, 1)
    
    
    def evaluate_posterior(self, ):
        return
    
    
    
    def sample_posterior(self, prng_state, x0, timesteps, jitter):
        """
        """
        
        if self.dynamics_function.self.RFF_num_feats > 0:  # RFF prior sampling
            return
        else:  # autoregressive sampling using conditionals
            return
        
        
        
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
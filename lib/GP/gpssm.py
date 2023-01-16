import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

from ..base import module
from .base import GP, SSM


class LGSSM(SSM):
    """
    Linear Gaussian State Space Model

    Temporal multi-output kernels have separate latent processes that can be coupled.
    Spatiotemporal kernel modifies the process noise across latent processes, but dynamics uncoupled.
    Multi-output GPs generally mix latent processes via dynamics as well.
    """

    As: jnp.ndarray
    Qs: jnp.ndarray
    H: jnp.ndarray
    m0: jnp.ndarray
    P0: jnp.ndarray

    def __init__(
        self, As, Qs, H, m0, P0, site_obs, site_Lcov, array_type=jnp.float32
    ):
        """
        :param jnp.ndarray As: transitions of shape (time, out, sd, sd)
        :param jnp.ndarray Qs: process noises of shape (time, out, sd, sd)
        :param jnp.ndarray site_locs: means of shape (time, out, 1)
        :param jnp.ndarray site_obs: means of shape (time, out, 1)
        :param jnp.ndarray site_Lcov: covariances of shape (time, out, out)
        """
        super().__init__(None, site_obs, site_Lcov, array_type)
        self.As = self._to_jax(As)
        self.Qs = self._to_jax(Qs)
        self.H = self._to_jax(H)
        self.m0 = self._to_jax(m0)
        self.P0 = self._to_jax(P0)

    def get_LDS(self):
        # convenience boundary transitions for kalman filter and smoother
        Id = jnp.eye(self.As.shape[-1])
        Zs = jnp.zeros_like(Id)
        As = jnp.concatenate((Id[None, ...], self.As, Id[None, ...]), axis=0)
        Qs = jnp.concatenate((Zs[None, ...], self.Qs, Zs[None, ...]), axis=0)
        return self.H, self.m0, self.P0, As, Qs

    ### posterior ###
    def entropy_posterior(self):
        """
        Precision of joint process is block-tridiagonal

        Compute the KL divergence and variational expectation of prior
        """
        return

    def evaluate_posterior(self, mean_only, compute_KL, compute_joint, jitter):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (locs,)
        :return:
            means of shape (time, out_dims, 1)
            covariances of shape (time, out_dims, out_dims)
        """
        H, m0, P0, As, Qs = self.get_LDS()

        smoother_means, smoother_covs, gains, logZ = fixed_interval_smoothing(
            H,
            m0,
            P0,
            As,
            Qs,
            self.site_obs,
            self.site_Lcov,
            compute_KL,
            compute_joint,
        )

        post_means = H @ smoother_means
        if compute_KL:  # compute using pseudo likelihood
            post_covs = H @ smoother_covs @ H.T
        post_covs = None if mean_only else H @ smoother_covs @ H.T

        if compute_KL:
            site_log_lik = pseudo_log_likelihood(
                post_means, post_covs, site_obs, site_Lcov
            )
            KL = site_log_lik - logZ

        else:
            KL = 0.0

        if compute_joint:
            cross_cov = gain[ind] @ post_cov[ind + 1]
            post_joint_covs = jnp.block(
                [[post_cov[ind], cross_cov], [cross_cov.T, post_cov[ind + 1]]]
            )

            return post_means, post_covs, post_joint_covs, KL

        else:  # only marginal
            return post_means, post_covs, KL

    ### sample ###
    def sample_prior(self, prng_state, num_samps, jitter):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param t: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            f_sample: the prior samples [S, N_samp]
        """
        H, m0, P0, As, Qs = self.get_LDS()
        return sample_LGSSM(
            H, m0, P0, As, Qs, prng_state, num_samps, jitter
        )  # (time, tr, state_dims, 1)

    def sample_posterior(
        self,
        prng_state,
        num_samps,
        jitter,
        compute_KL,
    ):
        """
        Sample from the posterior at specified time locations.
        Posterior sampling works by smoothing samples from the prior using the approximate Gaussian likelihood
        model given by the pseudo-likelihood, 𝓝(f|μ*,σ²*), computed during training.
         - draw samples (f*) from the prior
         - add Gaussian noise to the prior samples using auxillary model p(y*|f*) = 𝓝(y*|f*,σ²*)
         - smooth the samples by computing the posterior p(f*|y*)
         - posterior samples = prior samples + smoothed samples + posterior mean
                             = f* + E[p(f*|y*)] + E[p(f|y)]
        See Arnaud Doucet's note "A Note on Efficient Conditional Simulation of Gaussian Distributions" for details.

        :param X: the sampling input locations [N, 1]
        :param num_samps: the number of samples to draw [scalar]
        :param seed: the random seed for sampling
        :return:
            the posterior samples (eval_locs, num_samps, N, 1)
        """
        H, m0, P0, As, Qs = self.get_LDS()

        # sample prior at obs and eval locs
        prng_keys = jr.split(prng_state, 2)
        prior_samps = self.sample_prior(prng_keys[0], num_samps, t_all, jitter)

        # posterior mean
        post_means, _, KL_ss = fixed_interval_smoothing(
            H,
            m0,
            P0,
            As,
            Qs,
            self.site_obs,
            self.site_Lcov,
            interp_sites,
            mean_only=True,
            compute_KL=compute_KL,
            return_gains=False,
        )  # (time, N, 1)

        # noisy prior samples at eval locs
        prior_samps_t = prior_samps[site_ind, ...]
        prior_samps_eval = prior_samps[eval_ind, ...]

        prior_samps_noisy = prior_samps_t + self.site_Lcov[:, None, ...] @ jr.normal(
            prng_keys[1], shape=prior_samps_t.shape
        )  # (time, tr, N, 1)

        # smooth noisy samples
        def smooth_prior_sample(prior_samp_i):
            smoothed_sample, _, _, _ = fixed_interval_smoothing(
                H,
                m0,
                P0,
                As,
                Qs,
                prior_samp_i,
                self.site_Lcov,
                compute_KL=False,
                return_gains=False,
            )

            return smoothed_sample

        smoothed_samps = vmap(smooth_prior_sample, 1, 1)(prior_samps_noisy)

        # Matheron's rule pathwise samplig
        return prior_samps_eval - smoothed_samps + post_means[:, None, ...], KL_ss


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

    def __init__(
        self,
        dynamics_function,
        chol_process_noise,
        p0_mean,
        p0_Lcov,
        state_posterior,
        array_type=jnp.float32,
    ):
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

    def sample_prior(self, prng_state, num_samps, timesteps, jitter, x_eval):
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
                    prng_prior, x[:, None, None, :], jitter
                )  # (samp, x_dims, 1)
                noise = self.chol_process_noise[None, ...] @ jr.normal(
                    prng_state, shape=(num_samps, x_dims, 1)
                )
                x = x + fx[..., 0] + noise[..., 0]

                return (x, prng_prior), x

            x0 = (
                self.p0_Lcov[None, ...]
                @ jr.normal(prng_state, shape=(num_samps, x_dims, 1))
            )[
                ..., 0
            ]  # (num_samps, x_dims)
            prng_state, _ = jr.split(prng_state)
            _, x_samples = lax.scan(step, init=(x0, prng_state), xs=procnoise_keys)
            
            if x_eval is not None:
                eval_locs = x_eval[None, None, ...].repeat(num_samps, axis=0)
                f_samples = self.dynamics_function.sample_prior(
                    prng_state, eval_locs, jitter
                )  # (num_samps, x_dims, eval_locs)
            else:
                f_samples = None

        else:  # autoregressive sampling using conditionals

            def step(carry, inputs):
                x, x_obs, f_obs = carry  # (1, num_samps, x_dims)
                prng_state = inputs

                x = x[None, ..., 0]

                qf_m, qf_v = self.dynamics_function.evaluate_conditional(
                    x, x_obs, f_obs, mean_only=False, diag_cov=True, jitter=1e-6
                )

                fx = qf_m + qf_v @ jr.normal(
                    prng_state, shape=(num_samps, x_dims, 1)
                )  # (out_dims, num_samps, 1)
                prng_state, _ = jr.split(prng_state)

                x_obs = jnp.concatenate((x_obs, x), axis=1)  # (out_dims, obs_pts, 1)
                f_obs = jnp.concatenate((f_obs, fx), axis=1)  # (out_dims, obs_pts, 1)

                noise = self.chol_process_noise[None, ...] @ jr.normal(
                    prng_state, shape=(num_samps, x_dims, 1)
                )
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
            # _, x_samples = lax.scan(step, init=x0, xs=procnoise_keys)

        return x_samples.transpose(1, 0, 2), f_samples.transpose(0, 2, 1)  # (num_samps, time, state_dims)

    def evaluate_posterior(
        self,
    ):
        """
        The augmented KL divergence includes terms due to the state-space mapping [1]
        
        [1] Variational Gaussian Process State Space Models
        """
        # use method to obtain filter-smoother for q(x)
        
        # compute ELBO
        
        return post_mean, post_cov, aug_KL

    def sample_posterior(self, prng_state, num_samps, timesteps, jitter, x_eval):
        """ """
        x_dims = self.dynamics_function.kernel.in_dims

        eps_I = jitter * jnp.eye(x_dims)

        prng_states = jr.split(prng_state, 1 + timesteps)  # (num_samps, 2)
        prng_state, procnoise_keys = prng_states[0], prng_states[1:]

        if self.dynamics_function.RFF_num_feats > 0:  # RFF pathwise sampling
            ### TODO: more efficient implementation that avoids recomputing intermdiates
            def step(carry, inputs):
                x, prng_prior = carry  # (num_samps, x_dims)
                prng_state = inputs

                fx, _ = self.dynamics_function.sample_posterior(
                    prng_prior, x[:, None, None, :], jitter, compute_KL=False
                )  # (samp, x_dims, 1)
                noise = self.chol_process_noise[None, ...] @ jr.normal(
                    prng_state, shape=(num_samps, x_dims, 1)
                )
                x = x + fx[..., 0] + noise[..., 0]

                return (x, prng_prior), x

            x0 = (
                self.p0_Lcov[None, ...]
                @ jr.normal(prng_state, shape=(num_samps, x_dims, 1))
            )[
                ..., 0
            ]  # (num_samps, x_dims)
            prng_state, _ = jr.split(prng_state)
            _, x_samples = lax.scan(step, init=(x0, prng_state), xs=procnoise_keys)
            x_samples = x_samples.transpose(1, 0, 2)  # (num_samps, time, state_dims)
            
            if x_eval is not None:
                eval_locs = x_eval[None, None, ...].repeat(num_samps, axis=0)
                f_samples, _ = self.dynamics_function.sample_posterior(
                    prng_state, eval_locs, jitter, compute_KL=False
                )  # (num_samps, x_dims, eval_locs)
                f_samples = f_samples.transpose(0, 2, 1)  # (num_samps, eval_locs, state_dims)
                
            else:
                f_samples = None

        else:  # autoregressive sampling using conditionals
            return

        return x_samples, f_samples#, KL_f


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



class CTGPSSM(module):
    """
    Continuous-time Gaussian process state-space model
    """

    dynamics_function: GP
    chol_process_noise: jnp.ndarray  # (out_dims, state_dims, state_dims)

    p0_mean: jnp.ndarray  # (state_dims, 1)
    p0_Lcov: jnp.ndarray  # (state_dims, state_dims)

    state_posterior: LGSSM

    def __init__(
        self,
        dynamics_function,
        chol_process_noise,
        p0_mean,
        p0_Lcov,
        state_posterior,
        array_type=jnp.float32,
    ):
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

    def sample_prior(self, prng_state, num_samps, t_eval, jitter, x_eval):
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
                    prng_prior, x[:, None, None, :], jitter
                )  # (samp, x_dims, 1)
                noise = self.chol_process_noise[None, ...] @ jr.normal(
                    prng_state, shape=(num_samps, x_dims, 1)
                )
                x = x + fx[..., 0] + noise[..., 0]

                return (x, prng_prior), x

            x0 = (
                self.p0_Lcov[None, ...]
                @ jr.normal(prng_state, shape=(num_samps, x_dims, 1))
            )[
                ..., 0
            ]  # (num_samps, x_dims)
            prng_state, _ = jr.split(prng_state)
            _, x_samples = lax.scan(step, init=(x0, prng_state), xs=procnoise_keys)
            
            if x_eval is not None:
                eval_locs = x_eval[None, None, ...].repeat(num_samps, axis=0)
                f_samples = self.dynamics_function.sample_prior(
                    prng_state, eval_locs, jitter
                )  # (num_samps, x_dims, eval_locs)
            else:
                f_samples = None

        else:  # autoregressive sampling using conditionals

            def step(carry, inputs):
                x, x_obs, f_obs = carry  # (1, num_samps, x_dims)
                prng_state = inputs

                x = x[None, ..., 0]

                qf_m, qf_v = self.dynamics_function.evaluate_conditional(
                    x, x_obs, f_obs, mean_only=False, diag_cov=True, jitter=1e-6
                )

                fx = qf_m + qf_v @ jr.normal(
                    prng_state, shape=(num_samps, x_dims, 1)
                )  # (out_dims, num_samps, 1)
                prng_state, _ = jr.split(prng_state)

                x_obs = jnp.concatenate((x_obs, x), axis=1)  # (out_dims, obs_pts, 1)
                f_obs = jnp.concatenate((f_obs, fx), axis=1)  # (out_dims, obs_pts, 1)

                noise = self.chol_process_noise[None, ...] @ jr.normal(
                    prng_state, shape=(num_samps, x_dims, 1)
                )
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
            # _, x_samples = lax.scan(step, init=x0, xs=procnoise_keys)

        return x_samples.transpose(1, 0, 2), f_samples.transpose(0, 2, 1)  # (num_samps, time, state_dims)

    def evaluate_posterior(
        self,
    ):
        """
        The augmented KL divergence includes terms due to the state-space mapping [1]
        
        [1] Variational Gaussian Process State Space Models
        """
        # use method to obtain filter-smoother for q(x)
        
        # compute ELBO
        
        return post_mean, post_cov, aug_KL

    def sample_posterior(self, prng_state, num_samps, t_eval, jitter, x_eval):
        """ """
        x_dims = self.dynamics_function.kernel.in_dims

        eps_I = jitter * jnp.eye(x_dims)

        prng_states = jr.split(prng_state, 1 + timesteps)  # (num_samps, 2)
        prng_state, procnoise_keys = prng_states[0], prng_states[1:]

        if self.dynamics_function.RFF_num_feats > 0:  # RFF pathwise sampling
            ### TODO: more efficient implementation that avoids recomputing intermdiates
            def step(carry, inputs):
                x, prng_prior = carry  # (num_samps, x_dims)
                prng_state = inputs

                fx, _ = self.dynamics_function.sample_posterior(
                    prng_prior, x[:, None, None, :], jitter, compute_KL=False
                )  # (samp, x_dims, 1)
                noise = self.chol_process_noise[None, ...] @ jr.normal(
                    prng_state, shape=(num_samps, x_dims, 1)
                )
                x = x + fx[..., 0] + noise[..., 0]

                return (x, prng_prior), x

            x0 = (
                self.p0_Lcov[None, ...]
                @ jr.normal(prng_state, shape=(num_samps, x_dims, 1))
            )[
                ..., 0
            ]  # (num_samps, x_dims)
            prng_state, _ = jr.split(prng_state)
            _, x_samples = lax.scan(step, init=(x0, prng_state), xs=procnoise_keys)
            x_samples = x_samples.transpose(1, 0, 2)  # (num_samps, time, state_dims)
            
            if x_eval is not None:
                eval_locs = x_eval[None, None, ...].repeat(num_samps, axis=0)
                f_samples, _ = self.dynamics_function.sample_posterior(
                    prng_state, eval_locs, jitter, compute_KL=False
                )  # (num_samps, x_dims, eval_locs)
                f_samples = f_samples.transpose(0, 2, 1)  # (num_samps, eval_locs, state_dims)
                
            else:
                f_samples = None

        else:  # autoregressive sampling using conditionals
            return

        return x_samples, f_samples, KL_f
    
    


"""
def midpoint_step(f, x, dt):
    k1 = f(x)
    return f(x + k1*dt/2.)*dt

def Ralston_step(f, x, dt):
    k1 = f(x)
    return (k1/4. + (3./4.) * f(x + (2./3.) * k1*dt))*dt

def Heun_step(f, x, dt):
    k1 = f(x)
    return (k1 + f(x + k1*dt))*dt/2.

def RK4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + k1*dt/2.)
    k3 = f(x + k2*dt/2.)
    k4 = f(x + k3*dt)
    return (k1 + k4 + 2.*(k2 + k3)) * dt/6.

def RK38_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + k1*dt/3.)
    k3 = f(x + (k2 - k1/3.)*dt)
    k4 = f(x + (k3 - k2 + k1)*dt)
    return (k1 + k4 + 3.*(k2 + k3)) * dt/8.
"""
def Euler_coeffs():
    order = 1
    coeff = jnp.array([1.])
    q_coeff = None
    
    return order, coeff, q_coeff



def Ralston_coeffs():
    order = 2
    coeff = jnp.array([1./4., 3./4.])
    q_coeff = jnp.array([2./3.])
    
    return order, coeff, q_coeff



def RK4_coeffs():
    order = 4
    coeff = jnp.array([1./6., 1./3., 1./3., 1./6.])
    q_coeff = jnp.array([1./2., 1./2., 1.])

    return order, coeff, q_coeff



### integrators ###
def stepper(self, params, q, loss, I_e, dt, t, solver_id, compute_loss):
    """
    #dq_vh, dloss = self.integrator[integrator_id](q, I_e, dt, compute_loss)
    """
    order, coeff, q_coeff = self.solver_coeff[solver_id]

    neuron_model = self.neuron_model
    loss_rate = self.loss_rate

    npd = params['od']['neurons']
    lrd = params['od']['loss_rate']
    lrbuf = params['buf']['loss_rate']
    tau_s = params['eb']['neurons']['tau_s'][None, :, None]

    dloss = 0.

    ### Butcher tableau ###
    for k_ind in range(order): # unroll in JIT
        if k_ind > 0:
            k = jnp.concatenate(
                (f_vh_, 
                 -q_[..., -2:]/tau_s
                ), axis=-1)
            q_ = q.at[...].add(q_coeff[k_ind-1]*k*dt)
            f_vh_ = neuron_model.f_vh(npd, q_, I_e)

            # differences
            dq_vh = dq_vh.at[...].add(coeff[k_ind]*f_vh_)

        else: # first step alone is Euler step
            q_ = q
            f_vh_ = neuron_model.f_vh(npd, q_, I_e)

            # differences
            dq_vh = coeff[k_ind]*f_vh_

        ### updates ###
        if compute_loss and loss_rate is not None:
            q_vs_ = q_[..., jnp.array([0, -2])]
            l = loss_rate.l(lrbuf, lrd, q_vs_, t).sum()
            dloss += coeff[k_ind]*l

    ### loss rate computation ###    
    if compute_loss:
        loss += dloss*dt

    ### evolve state ###
    fac = jnp.exp(-dt/tau_s)
    q = q.at[..., :-2].add(dq_vh*dt)
    q = q.at[..., -2:].multiply(fac)
    return q, loss


def stepper_aug(self, params, grads, q_aug, loss, I_e_vjp_fun, I_e, dt, t, 
                solver_id, compute_loss, state_output):
    """
    Adjoint dynamics, also gives f(q) for forward dynamics
    q_aug is (q, lambda)
    """
    order, coeff, q_coeff = self.solver_coeff[solver_id]

    q_d = self.neuron_model.q_d
    neuron_model = self.neuron_model
    loss_rate = self.loss_rate

    npd = params['od']['neurons']
    lrd = params['od']['loss_rate']
    lrbuf = params['buf']['loss_rate']
    tau_s = params['eb']['neurons']['tau_s'][None, :, None]

    if loss_rate is not None:
        fl = lambda b, x, y: loss_rate.l(b, x, y, t).sum() # sum is mean over trials (1/N factor inside)

    dloss = 0.

    ### Butcher tableau ###
    for k_ind in range(order): # unroll in JIT
        if k_ind > 0:
            k = jnp.concatenate(
                (f_vh_, 
                 -q_[..., -2:]/tau_s, 
                 -lmb_vh_dfdq[..., :1] + dl_dqvs[..., :1], 
                 -lmb_vh_dfdq[..., 1:-2], 
                 lmb_[..., -2:-1]/tau_s + dl_dqvs[..., 1:], 
                 lmb_[..., -1:]/tau_s - lmb_vh_dfdq[..., -1:]
                ), axis=-1)
            q_aug_ = q_aug.at[...].add(q_coeff[k_ind-1]*k*dt)
            q_ = q_aug_[..., :q_d]
            lmb_ = q_aug_[..., q_d:]

            f_vh_, vjp_fun = vjp(neuron_model.f_vh, npd, q_, I_e) # compute VJPs
            lmb_vh_ = lmb_[..., :-2]
            vjps = vjp_fun(lmb_vh_)
            lmb_vh_dfdq = vjps[1]
            lmb_vh_dfdIe = vjps[2]

            grad_scale = coeff[k_ind]*dt

            lmb_vh_dfdp = vjps[0] # shape of param_dict
            add_to_grad_dict(grads['od']['neurons'], lmb_vh_dfdp, grad_scale) # accumulate directly into parameter tree

            if loss_rate is not None:
                q_vs_ = q_[..., jnp.array([0, -2])] # shape (tr, N, dims)
                l, (dl_dqvs,) = value_and_grad(fl, (2,))(lrbuf, lrd, q_vs_)
                #dl_dp = grad(fl, 0)(lrd, q_vs_)
                if compute_loss:
                    dloss += coeff[k_ind]*l
                #add_to_grad_dict(grads['od']['loss_rate'], dl_dp, grad_scale) # accumulate directly into parameter tree
            else:
                dl_dqvs = jnp.zeros_like(q_[..., jnp.array([0, -2])])

            # differences
            dq_vh = dq_vh.at[...].add(coeff[k_ind]*f_vh_)
            dlmb_vh = dlmb_vh.at[...].add(
                coeff[k_ind]*(-lmb_vh_dfdq[..., :-2]).at[..., 0].add(dl_dqvs[..., 0]))
            inhom_lmb_sI = inhom_lmb_sI.at[...].add(
                coeff[k_ind]*jnp.stack((dl_dqvs[..., 1], -lmb_vh_dfdq[..., -1]), axis=-1))

        else: # first step alone is Euler step
            q_ = q_aug[..., :q_d]
            lmb_ = q_aug[..., q_d:]

            f_vh_, vjp_fun = vjp(neuron_model.f_vh, npd, q_, I_e) # compute VJPs
            lmb_vh_ = lmb_[..., :-2]
            vjps = vjp_fun(lmb_vh_)
            lmb_vh_dfdq = vjps[1]
            lmb_vh_dfdIe = vjps[2]

            grad_scale = coeff[k_ind]*dt

            lmb_vh_dfdp = vjps[0] # shape of param_dict
            add_to_grad_dict(grads['od']['neurons'], lmb_vh_dfdp, grad_scale) # accumulate directly into parameter tree

            if loss_rate is not None:
                q_vs_ = q_[..., jnp.array([0, -2])] # shape (tr, N, dims)
                l, (dl_dqvs,) = value_and_grad(fl, (2,))(lrbuf, lrd, q_vs_)
                #dl_dp = grad(fl, 0)(lrd, q_vs_)
                if compute_loss:
                    dloss += coeff[k_ind]*l
                #add_to_grad_dict(grads['od']['loss_rate'], dl_dp, grad_scale) # accumulate directly into parameter tree
            else:
                dl_dqvs = jnp.zeros_like(q_[..., jnp.array([0, -2])])

            # differences
            dq_vh = coeff[k_ind]*f_vh_
            dlmb_vh = coeff[k_ind]*(-lmb_vh_dfdq[..., :-2]).at[..., 0].add(dl_dqvs[..., 0])
            inhom_lmb_sI = coeff[k_ind]*jnp.stack((dl_dqvs[..., 1], -lmb_vh_dfdq[..., -1]), axis=-1)


        ### accumulate continuous gradients (use lambda^-) ###
        if I_e_vjp_fun is not None: # input object grads
            out = I_e_vjp_fun(lmb_vh_dfdIe) # TODO: evaluate outside for loop as I_e doesn't change (stepwise constant)
            add_to_grad_dict(grads['od']['inputs'], out[0], grad_scale)

        if grads['eb']['neurons']['tau_s'].shape != (0,):
            dgrad_tau_s = (lmb_[..., -2:]*q_[..., -2:]).sum(-1).sum(0)/tau_s[0, :, 0]**2 * grad_scale
            if tau_s.shape[1] == 1: # sum over neurons
                dgrad_tau_s = dgrad_tau_s.sum(0, keepdims=True)
            grads['eb']['neurons']['tau_s'] = grads['eb']['neurons']['tau_s'].at[...].add(
                dgrad_tau_s)


    ### loss rate computation ###
    if compute_loss: # note we integrate backward so minus sign
        loss -= dloss*dt


    ### evolve state ###
    fac_adj = jnp.exp(dt/tau_s)
    q_aug = q_aug.at[..., q_d:-2].add(dlmb_vh*dt)
    q_aug = q_aug.at[..., -2:].multiply(fac_adj)
    q_aug = q_aug.at[..., -2:].add(tau_s*(fac_adj-1.) * inhom_lmb_sI)

    if state_output is not None: # q^+ stored trajectories
        q_aug = q_aug.at[..., :q_d].set(state_output[t, ...])
    else:
        fac = jnp.exp(-dt/tau_s)
        q_aug = q_aug.at[..., :q_d-2].add(dq_vh*dt)
        q_aug = q_aug.at[..., q_d-2:q_d].multiply(fac)

    return q_aug, loss, grads
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

from ..base import module
from .base import SSM
from .kernels import MarkovianKernel
from .markovian import LGSSM


# HMM
class CTHMM(module):
    """
    Continuous-time finite discrete state with Markovian transition dynamics
    """

    pre_T: jnp.ndarray  # pre-softmax

    fixed_grid_locs: bool
    site_locs: jnp.ndarray
    site_ll: jnp.ndarray  # pseudo-observation log probabilities
    K: int  # state dimension

    def __init__(self, pre_T, site_ll):
        """
        :param jnp.ndarray T: transition matrix of shape (K, K)
        :param jnp.ndarray site_ll: pseudo-observation log probabilities (ts, K)
        """
        self.pre_T = pre_T

        self.fixed_grid_locs = fixed_grid_locs
        self.site_locs = site_locs
        self.site_ll = site_ll
        self.K = pre_T.shape[-1]

    def compute_stationary_distribution(self):
        """
        Solve overparameterized augmented linear equation via QR decomposition
        """
        T = jax.nn.softmax(self.pre_T, axis=0)

        P = T - jnp.eye(self.K)
        A = jnp.concatenate((P, jnp.ones(self.K)[None, :]), axis=0)
        b = jnp.zeros(self.K + 1).at[-1].set(1.0)

        # Q, R = jnp.linalg.qr(A)
        # jax.scipy.linalg.solve_triangular(R, jnp.Q.T @ b)
        pi = jnp.linalg.pinv(A) @ b  # via pinv
        return pi

    def filter_observations(self, log_p0, obs_ll, reverse, compute_marginal):
        """
        Filtering observations with discrete variable, message passing to marginalize efficiently.

        :param jnp.ndarray log_p0: initial probability of shape (K,)
        :param jnp.ndarray obs_ll: observation log likelihoods per latent state of shape (ts, K)
        """
        log_T = jax.nn.log_softmax(self.pre_T, axis=0)

        def step(carry, inputs):
            log_p_ahead, log_marg = carry  # log p(z_t | x_{<t})
            ll = inputs

            log_normalizer = jax.nn.logsumexp(log_p_ahead + ll)  # log p(x_t | x_{<t})
            if compute_marginal:
                log_marg += log_normalizer

            # log p(z_t | x_{<=t}) = log[ p(z_t | x_{<t}) * p(x_t|z_t) / p(x_t | x_{<t}) ]
            log_p_at = (log_p_ahead + ll) - log_normalizer  # (K,)

            # log[ \sum_{z_t} p(z_t|x_{<=t}) * p(z_{t+1}|z_t) ]
            log_p_ahead = jax.nn.logsumexp(log_p_at[None, :] + log_T, axis=-1)  # (K,)

            return (log_p_ahead, log_marg), log_p_at

        (_, log_marg), log_p_ats = lax.scan(
            step, init=(log_p0, 0.0), xs=obs_ll, reverse=reverse
        )
        return log_p_ats, log_marg

    def evaluate_posterior(self, obs_ll, compute_KL):
        """
        Forward-backward algorithm

        :param jnp.array obs_ll: observation log likelihoods of shape (ts, K)
        :returns:
            means of shape (out_dims, num_samps, time, 1)
            covariances of shape (out_dims, num_samps, time, time)
        """
        pi = self.compute_stationary_distribution()
        log_pi = jnp.log(jnp.maximum(pi, 1e-12))
        log_T = jax.nn.log_softmax(self.pre_T, axis=0)  # transition matrix

        log_betas, log_marg = self.filter_observations(
            log_pi, obs_ll, True, compute_KL
        )  # backward

        # combine forward with backward sweep for marginals
        def step(carry, inputs):
            log_alpha, var_expect = carry  # log p(z_t | x_{<t})
            log_beta, ll = inputs

            log_post = log_alpha + log_beta  # log[ p(z_t | x_{<t}) * p(z_t | x_{>=t})]
            if compute_KL:
                var_expect += (jnp.exp(log_post) * ll).sum()

            # log p(z_t | x_{<=t}) = log[ p(z_t | x_{<t}) * p(x_t|z_t) / p(x_{<=t}) ]
            log_p_at = jax.nn.log_softmax(log_alpha + ll)  # (K,)
            # log[ \sum_{z_t} p(z_t|x_{<=t}) * p(z_{t+1}|z_t) ]
            log_alpha = jax.nn.logsumexp(log_p_at[None, :] + log_T, axis=-1)  # (K,)

            return (log_alpha, var_expect), log_post

        (_, var_expect), log_posts = lax.scan(
            step, init=(log_pi, 0.0), xs=(log_betas, obs_ll), reverse=False
        )  # (ts, K)

        KL = log_marg - var_expect
        aux = (log_marg, log_betas, log_pi, log_T)
        return log_posts, KL, aux

    def sample_prior(self, prng_state, num_samps, timesteps):
        """
        Sample from the HMM latent variables.

        :param jnp.ndarray p0: initial probabilities of shape (num_samps, state_dims)
        :returns: state tensor of shape (trials, time)
        :rtype: torch.tensor
        """
        pi = self.compute_stationary_distribution()
        log_pi = jnp.log(jnp.maximum(pi, 1e-12))

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)
        log_T = jax.nn.log_softmax(self.pre_T, axis=0)  # transition matrix

        def step(carry, inputs):
            log_p_cond = carry  # log p(z_t | z_{<t})
            prng_state = inputs

            s = jr.categorical(prng_state, log_p_cond)  # (K,)
            log_p_cond = log_T[:, s] + log_p_cond[s]  # next step probabilities
            return log_p_cond, s

        def sample_i(prng_state):
            prng_keys = jr.split(prng_state, timesteps)
            _, states = lax.scan(step, init=log_pi, xs=prng_keys)
            return states

        states = vmap(sample_i, 0, 1)(prng_states)
        return states  # (time, tr, K)

    def sample_posterior(self, prng_state, num_samps, timesteps, compute_KL):
        """
        Forward-backward sampling algorithm

        :param jnp.array x: input of shape (time, num_samps, in_dims, 1)
        :returns:
            means of shape (out_dims, num_samps, time, 1)
            covariances of shape (out_dims, num_samps, time, time)
        """
        log_posts, KL, aux = self.evaluate_posterior(self.site_ll, compute_KL)
        log_marg, log_betas, log_pi, log_T = aux

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        # combine forward with backward sweep
        def step(carry, inputs):
            log_p_cond = carry  # log p(z_t | z_{<t}, x_{1:T})
            log_beta, prng_state = inputs

            log_p = log_beta + log_p_cond
            s = jr.categorical(prng_state, log_p_cond)  # (K,)
            log_p_cond = log_T[:, s] + log_p_cond  # next step probabilities

            return log_p_cond, s

        def sample_i(prng_state):
            prng_keys = jr.split(prng_state, timesteps)
            _, states = lax.scan(
                step, init=jnp.zeros(self.K), xs=(log_betas, prng_keys), reverse=False
            )  # (ts, K)
            return states

        states = vmap(sample_i, 0, 1)(prng_states)  # (time, tr, K)
        return states, KL


### switch states ###
class Transition(module):
    """
    Transition matrices
    """

    A: jnp.ndarray
    Q: jnp.ndarray

    def __init__(self, A, Q):
        """
        :parma jnp.ndarray A: transition matrices (num_switches, sd, sd)
        :parma jnp.ndarray Q: process noise matrices (num_switches, sd, sd)
        """
        super().__init__()
        self.A = A
        self.Q = Q


class WhiteNoiseSwitch(Transition):
    """
    White noise processes at different orders
    """

    def __init__(self, N, R, B, Lam):
        out_dims = B.shape[-1]
        state_dims = N.shape[0]
        in_dims = 1
        super().__init__(in_dims, out_dims, state_dims)
        self.N = N
        self.R = R
        self.B = B
        self.Lam = Lam

    @eqx.filter_vmap()
    def _state_dynamics(self):
        G, Q = self.parameterize()
        F = -G / 2.0
        L = []
        return F, L, Q

    @eqx.filter_vmap()
    def _state_output(self):
        """
        Pinf is the solution of the Lyapunov equation F P + P F^T + L Qc L^T = 0
        Pinf = solve_continuous_lyapunov(F, Q)
        In this parameterization Pinf is just the identity
        """
        minf = jnp.zeros((state_dims,))
        Pinf = jnp.eye(self.state_dims)
        return self.H, minf, Pinf

    @eqx.filter_vmap(kwargs=dict(dt=0))
    def _state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-7/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [4, 4]
        """
        G, _ = self.parameterize()
        return expm(-dt * G / 2)


# switching LGSSM
class SwitchingLTI(SSM):
    """
    In the switching formulation, every observation point is doubled to account
    for infinitesimal switches inbetween
    """

    switch_logits: jnp.ndarray

    CTHMM: module
    switch_transitions: Transition
    markov_kernel_group: MarkovianKernel

    def __init__(
        self,
        CTHMM,
        switch_logits,
        markov_kernel_group,
        switch_transitions,
        site_locs,
        site_obs,
        site_Lcov,
        fixed_grid_locs=False,
    ):
        """
        :param jnp.ndarray switch_logits: matrix of logit vectors for event at switch (K, K, switches)
        """
        super().__init__(site_locs, site_obs, site_Lcov)
        self.CTHMM = CTHMM
        self.markov_kernel_group = markov_kernel_group
        self.switch_transitions = switch_transitions

        self.fixed_grid_locs = fixed_grid_locs

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

    def get_conditional_LDS(self, dt, tsteps, s_path):
        """
        Insert switches
        """
        H, minf, Pinf, As, Qs = self.markov_kernel.get_LDS(dt, tsteps)

        Id = jnp.eye(self.As.shape[-1])
        Zs = jnp.zeros_like(Id)
        As = jnp.concatenate((Id[None, ...], self.As, Id[None, ...]), axis=0)
        Qs = jnp.concatenate((Zs[None, ...], self.Qs, Zs[None, ...]), axis=0)

        return self.H, self.minf, Pinf, As, Qs

    def get_site_params(self):
        """
        Insert unobserved locations
        """
        site_obs = self.site_obs
        site_Lcov = self.site_Lcov
        return site_obs, site_Lcov

    def sample_prior(self):
        """
        Sample from the joint prior
        """
        prng_state, num_samps, timesteps = self.CTHMM.sample_prior()

        H, minf, Pinf, As, Qs = self.get_conditional_LDS()

        return x_samples, s_samples

    def evaluate_conditional_posterior(self, s_path):
        """
        Compute posterior conditioned on HMM path
        """
        site_obs, site_Lcovs = self.get_site_params()
        H, minf, Pinf, As, Qs, site_obs, site_Lcov = self.get_conditional_LDS()
        return

    def sample_posterior(self):
        """
        Sample from joint posterior
        """
        return

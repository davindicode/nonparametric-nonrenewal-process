import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.scipy.linalg import expm

from ..base import ArrayTypes_, module
from ..utils.jax import safe_log

from .base import SSM
from .kernels import GroupMarkovian, MarkovianKernel
from .linalg import bdiag, evaluate_LGSSM_posterior, LTI_process_noise
from .markovian import interpolation_times, order_times, sample_LGSSM


def obs_switch_locs(obs_locs, switch_locs):
    """
    Temporally order observation and switch locations
    Return relative order indices
    """
    obs_num = len(obs_locs)
    sw_num = len(switch_locs)
    unique_locs = jnp.concatenate((switch_locs, obs_locs), axis=0)
    locs = jnp.concatenate((unique_locs, switch_locs), axis=0)  # double switch locs

    sort_ind = jnp.argsort(locs)
    locs = locs[sort_ind]
    argsort_ind = jnp.argsort(sort_ind)
    
    trans_inds, jump_inds = argsort_ind[sw_num:], jnp.argsort(sort_ind)[:sw_num]
    obs_inds = argsort_ind[sw_num:sw_num+obs_num]
    unique_locs = jnp.sort(unique_locs)
    
    return locs, trans_inds, jump_inds, obs_inds, unique_locs


def tile_between_inds(path, inds, ts):
    """
    Continue the GP state path around observation points
    """
    assert len(path) == len(inds) + 1

    def step(carry, inputs):
        i, p, j = carry
        ind = inputs

        i = lax.cond(ind >= j[i], lambda i: i + 1, lambda i: i, i)

        return (i, p, j), p[i]

    _, tiled_path = lax.scan(step, init=(0, path, inds), xs=jnp.arange(ts))
    return tiled_path  # (ts,)


### HMM ###
def markov_stationary_distribution(T, K):
    P = T - jnp.eye(K)
    A = jnp.concatenate((P, jnp.ones(K)[None, :]), axis=0)
    b = jnp.zeros(K + 1).at[-1].set(1.0)

    # Q, R = jnp.linalg.qr(A)
    # jax.scipy.linalg.solve_triangular(R, jnp.Q.T @ b)
    pi = jnp.linalg.pinv(A) @ b  # via pinv
    return pi


class DTMarkovChain(module):
    """
    Discrete-time finite discrete state with Markovian transition dynamics

    Regular grid of time points
    """

    pre_T: jnp.ndarray  # pre-softmax
    site_ll: jnp.ndarray  # pseudo-observation log probabilities
    K: int  # state dimension

    def __init__(self, pre_T, site_ll, array_type="float32"):
        """
        :param jnp.ndarray T: transition matrix of shape (K, K)
        :param jnp.ndarray site_ll: pseudo-observation log probabilities (ts, K)
        """
        super().__init__(array_type)
        self.pre_T = self._to_jax(pre_T)
        self.site_ll = self._to_jax(site_ll)
        self.K = pre_T.shape[-1]

    def compute_chain(self):
        """
        Solve overparameterized augmented linear equation via QR decomposition
        """
        log_T = jax.nn.log_softmax(self.pre_T, axis=0)
        pi = markov_stationary_distribution(jnp.exp(log_T), self.K)
        return pi, log_T

    def filter_observations(self, log_p0, obs_ll, reverse, compute_marginal):
        """
        Filtering observations with discrete variable, message passing to marginalize efficiently.

        :param jnp.ndarray log_p0: initial probability of shape (K,)
        :param jnp.ndarray obs_ll: observation log likelihoods per latent state of shape (ts, K)
        """
        pi, log_T = self.compute_chain()
        log_pi = safe_log(pi)

        def step(carry, inputs):
            log_p_ahead, log_marg = carry  # log p(z_t | x_{<t})
            ll = inputs

            log_normalizer = jax.nn.logsumexp(log_p_ahead + ll)  # log p(x_t | x_{<t})
            if compute_marginal:
                log_marg += log_normalizer

            # log p(z_t | x_{<=t}) = log[ p(z_t | x_{<t}) * p(x_t|z_t) / p(x_t | x_{<t}) ]
            log_p_at = (log_p_ahead + ll) - log_normalizer  # (K,)

            # log[ \sum_{z_t} p(z_t|x_{<=t}) * p(z_{t+1}|z_t) ]
            log_p_ahead = jax.nn.logsumexp(log_p_at[None, :] + log_T, axis=1)  # (K,)

            return (log_p_ahead, log_marg), log_p_at

        (_, log_marg), log_p_ats = lax.scan(
            step, init=(log_pi, 0.0), xs=obs_ll, reverse=reverse
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
        pi, log_T = self.compute_chain()
        log_pi = safe_log(pi)

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
            log_alpha = jax.nn.logsumexp(log_p_at[None, :] + log_T, axis=1)  # (K,)

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
        :returns:
            state tensor of shape (trials, time)
        """
        pi, log_T = self.compute_chain()
        log_pi = safe_log(pi)

        prng_states = jr.split(prng_state, num_samps * timesteps).reshape(
            num_samps, timesteps, -1
        )  # (num_samps, timesteps, 2)

        def step(carry, inputs):
            log_p_cond = carry  # log p(z_t | z_{<t})
            prng_state = inputs

            s = jr.categorical(prng_state, log_p_cond)  # 0 to K-1 integers
            log_p_cond = log_T[:, s]  # next step probabilities
            return log_p_cond, s

        def sample_i(prng_keys):
            _, states = lax.scan(step, init=log_pi, xs=prng_keys)
            return states

        states = vmap(sample_i)(prng_states)
        return states  # (tr, time)

    def sample_posterior(self, prng_state, num_samps, compute_KL, MAP=False):
        """
        Forward-backward sampling algorithm

        :param jnp.array x: input of shape (time, num_samps, in_dims, 1)
        :returns:
            means of shape (out_dims, num_samps, time, 1)
            covariances of shape (out_dims, num_samps, time, time)
        """
        timesteps = self.site_ll.shape[0]

        log_posts, KL, aux = self.evaluate_posterior(self.site_ll, compute_KL)
        log_marg, log_betas, log_pi, log_T = aux

        prng_states = jr.split(prng_state, num_samps * timesteps).reshape(
            num_samps, timesteps, -1
        )  # (num_samps, timesteps, 2)

        # combine forward with backward sweep
        def step(carry, inputs):
            log_p_cond = carry  # log p(z_t | z_{<t}, x_{1:T})
            log_beta, prng_state = inputs

            log_p = log_beta + log_p_cond
            if MAP:
                s = jnp.argmax(log_p)
            else:
                s = jr.categorical(prng_state, log_p)  # 0 to K-1 integers
            log_p_cond = log_T[:, s]  # next step probabilities

            return log_p_cond, s

        def sample_i(prng_keys):
            _, states = lax.scan(
                step, init=jnp.zeros(self.K), xs=(log_betas, prng_keys), reverse=False
            )  # (ts, K)
            return states

        states = vmap(sample_i)(prng_states)  # (tr, time)
        return states, KL


class CTMarkovChain(module):
    """
    Continuous-time finite discrete state with Markovian transition dynamics
    """

    Q_f: jnp.ndarray  # Q is Q_f with right column to make Q.sum(-1) = 0

    fixed_grid_locs: bool
    site_locs: jnp.ndarray
    site_ll: jnp.ndarray  # pseudo-observation log probabilities
    K: int  # state dimension

    def __init__(
        self, Q_f, site_locs, site_ll, fixed_grid_locs=False, array_type="float32"
    ):
        """
        :param jnp.ndarray Q_f: transition matrix of shape (K, K-1)
        :param jnp.ndarray site_ll: pseudo-observation log probabilities (ts, K)
        """
        super().__init__(array_type)
        self.Q_f = self._to_jax(Q_f)

        self.fixed_grid_locs = fixed_grid_locs
        self.site_locs = self._to_jax(site_locs)
        self.site_ll = self._to_jax(site_ll)
        self.K = pre_T.shape[-1]

    def get_site_locs(self):
        if self.fixed_grid_locs:
            locs = lax.stop_gradient(self.site_locs)  # (ts,)
            return locs, locs[1:2] - locs[0:1]
        else:
            return self.site_locs, jnp.diff(self.site_locs)

    def compute_chain(self, dt):
        """
        Solve overparameterized augmented linear equation via QR decomposition
        """
        Q = jnp.concatenate((self.Q_f, -self.Q_f.sum(1, keepdims=True)), axis=1)
        T = expm(dt[:, None, None] * Q[None, ...])  # (ts, sd, sd)
        pi = markov_stationary_distribution(T, self.K)  # limit delta t -> 0?
        return pi, log_T

    def filter_observations(self, log_p0, obs_ll, reverse, compute_marginal):
        """
        Filtering observations with discrete variable, message passing to marginalize efficiently.

        :param jnp.ndarray log_p0: initial probability of shape (K,)
        :param jnp.ndarray obs_ll: observation log likelihoods per latent state of shape (ts, K)
        """
        pi, log_T = self.compute_chain()
        log_pi = safe_log(pi)

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
            step, init=(log_pi, 0.0), xs=obs_ll, reverse=reverse
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
        pi, log_T = self.compute_chain()
        log_pi = safe_log(pi)

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
        pi, log_T = self.compute_chain()
        log_pi = safe_log(pi)

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        def step(carry, inputs):
            log_p_cond = carry  # log p(z_t | z_{<t})
            prng_state = inputs

            s = jr.categorical(prng_state, log_p_cond)  # 0 to K-1 integers
            log_p_cond = log_T[:, s]  # next step probabilities
            return log_p_cond, s

        def sample_i(prng_state):
            prng_keys = jr.split(prng_state, timesteps)
            _, states = lax.scan(step, init=log_pi, xs=prng_keys)
            return states

        states = vmap(sample_i, 0, 0)(prng_states)  # (tr, time)
        return states

    def sample_posterior(self, prng_state, num_samps, compute_KL, MAP=False):
        """
        Forward-backward sampling algorithm

        :param jnp.array x: input of shape (time, num_samps, in_dims, 1)
        :returns:
            means of shape (out_dims, num_samps, time, 1)
            covariances of shape (out_dims, num_samps, time, time)
        """
        timesteps = self.site_ll.shape[0]

        log_posts, KL, aux = self.evaluate_posterior(self.site_ll, compute_KL)
        log_marg, log_betas, log_pi, log_T = aux

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        # combine forward with backward sweep
        def step(carry, inputs):
            log_p_cond = carry  # log p(z_t | z_{<t}, x_{1:T})
            log_beta, prng_state = inputs

            log_p = log_beta + log_p_cond
            if MAP:
                s = jnp.argmax(log_p)
            else:
                s = jr.categorical(prng_state, log_p)  # 0 to K-1 integers
            log_p_cond = log_T[:, s]  # next step probabilities

            return log_p_cond, s

        def sample_i(prng_state):
            prng_keys = jr.split(prng_state, timesteps)
            _, states = lax.scan(
                step, init=jnp.zeros(self.K), xs=(log_betas, prng_keys), reverse=False
            )  # (ts, K)
            return states

        states = vmap(sample_i, 0, 0)(prng_states)  # (tr, time)
        return states, KL


### switch transitions ###
class Transition(module):
    """
    Transition matrices
    """

    state_dims: int
    out_dims: int

    def __init__(self, out_dims, state_dims, array_type):
        """
        :parma jnp.ndarray A: transition matrices (num_switches, sd, sd)
        :parma jnp.ndarray Q: process noise matrices (num_switches, sd, sd)
        """
        super().__init__(array_type)
        self.state_dims = state_dims
        self.out_dims = out_dims

    def state_transition(self):
        """
        Block diagonal from (out_dims, x_dims, x_dims) to (state_dims, state_dims)

        :param jnp.ndarray dt: time intervals of shape broadcastable with (out_dims,)
        """
        (
            A,
            Q,
        ) = (
            self._state_transition()
        )  # vmap over output dims, (out_dims, orders, sd, sd)
        return vmap(bdiag, 1, 0)(A), vmap(bdiag, 1, 0)(Q)  # (order, sd, sd)


class WhiteNoiseSwitch(Transition):
    """
    White noise processes at different orders
    Matches the diagonal stationary covariance of the Matern with corresponding order
    """

    noise_vars: jnp.ndarray

    def __init__(self, out_dims, state_dims, noise_vars, array_type="float32"):
        """
        :param jnp.ndarray noise_vars: (out_dims, orders)
        """
        super().__init__(out_dims, state_dims, array_type)
        self.noise_vars = self._to_jax(noise_vars)

    @eqx.filter_vmap()
    def _state_transition(self):
        """
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return:
            state transition matrix A (orders, sd, sd)
        """
        O = len(self.noise_vars)
        A = jnp.zeros((O, self.state_dims, self.state_dims))
        Q = jnp.zeros((O, self.state_dims, self.state_dims))
        for o in range(O + 1):  # unroll, last order is identity transition
            inds = jnp.arange(o)
            A = A.at[o, inds, inds].set(1.0)
            if o < O:
                Q = Q.at[o, o, o].set(self.noise_vars[o])

        return A, Q


### switching LGSSM ###
vLGSSM = vmap(
    evaluate_LGSSM_posterior,
    (None, None, None, 0, 0, None, None, None, None, None, None, None, None),
    (0, 0, 0),
)  # vmap over MC

vLGSSM_ = vmap(
    evaluate_LGSSM_posterior,
    (None, None, None, 0, 0, 0, None, None, None, None, None, None, None),
    (0, 0, 0),
)  # vmap over MC

vsample = vmap(sample_LGSSM, (None, None, None, 0, 0, 0, None, None), 0)  # vmap over MC


class JumpLTI(SSM):
    """
    In the switching formulation, every observation point is doubled to account
    for infinitesimal switches inbetween
    """

    switch_locs: jnp.ndarray
    switch_HMM: DTMarkovChain

    switch_transitions: Transition
    kernel: MarkovianKernel

    fixed_grid_locs: bool

    def __init__(
        self,
        switch_locs,
        switch_HMM,
        kernel,
        switch_transitions,
        site_locs,
        site_obs,
        site_Lcov,
        fixed_grid_locs=False,
    ):
        """
        :param jnp.ndarray switch_locs: locations of switches (K, K, switches)
        """
        assert switch_HMM.site_ll.shape[0] == switch_locs.shape[0]
        assert switch_HMM.array_type == kernel.array_type
        assert switch_HMM.array_type == switch_transitions.array_type
        super().__init__(
            site_locs, site_obs, site_Lcov, ArrayTypes_[switch_HMM.array_type]
        )
        self.switch_locs = self._to_jax(switch_locs)
        self.switch_HMM = switch_HMM

        self.switch_transitions = switch_transitions
        self.kernel = kernel

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

    ### site and switch locations ###
    def get_site_locs(self):
        """
        Both site obervation locs and switch locs contribute the site locations
        """
        locs, trans_inds, jump_inds, obs_inds, unique_locs = obs_switch_locs(
            self.site_locs, self.switch_locs
        )

        if self.fixed_grid_locs:
            locs = lax.stop_gradient(locs)  # (ts,)
            return locs, trans_inds, jump_inds, obs_inds, unique_locs[1:2] - unique_locs[0:1]
        else:
            return locs, trans_inds, jump_inds, obs_inds, jnp.diff(unique_locs)

    def get_site_params(self, obs_inds, ts):
        """
        Insert unobserved locations at switch edge locations
        
        :param int ts: total time points = len(obs_inds) + len(switch_inds)
        """
        sd = self.site_obs.shape[-2]

        all_obs = jnp.zeros((ts, sd, 1), dtype=self.array_dtype())
        all_Lcov = 1e6 * jnp.eye(sd, dtype=self.array_dtype())[None, ...].repeat(
            ts, axis=0
        )  # empty observations

        all_obs = all_obs.at[obs_inds].set(self.site_obs)
        all_Lcov = all_Lcov.at[obs_inds].set(self.site_Lcov)
        return all_obs, all_Lcov

    def get_conditional_LDS(self, dt, trans_inds, jump_inds, j_path):
        """
        Insert switches inbetween observation sites
        Each observation site has a state
        Each switch applies from A_{s}(t/2) * A_{ss'} * A_{s'}(t/2) with empty
        observation at the * locations

        :param jnp.ndarray s_path: GP state trajectories (switch_locs + 1,), [0, s_dims)
        :param jnp.ndarray j_path: switch state trajectories (switch_locs,), [0, j_dims)
        """
        sd = self.kernel.state_dims
        num_switch_locs = len(jump_inds)
        trans_without_jumps = len(trans_inds)  # (ts,)
        trans_with_jumps = num_switch_locs + trans_without_jumps + 1  # include boundary transitions
        
        num_samps = j_path.shape[0]
        As = jnp.empty((num_samps, trans_with_jumps, sd, sd), dtype=self.array_dtype())
        Qs = jnp.empty((num_samps, trans_with_jumps, sd, sd), dtype=self.array_dtype())

        # use out_dims of kernel as states
        H, Pinf, As_, Qs_ = self.kernel.get_LDS(dt, trans_without_jumps)  # (ts+1, sd, sd)

        # jump matrices
        array_indexing = lambda ind, array: array[ind, ...]
        varray_indexing = vmap(array_indexing, (0, None), 0)

        Ajs, Qjs = self.switch_transitions.state_transition()  # (states, sd, sd)
        Ajs, Qjs = varray_indexing(j_path, Ajs), varray_indexing(
            j_path, Qjs
        )  # (num_samps, ts, sd, sd)
        
        # insert matrices
        trans_inds = jnp.concatenate(
            (jnp.array([0]), trans_inds + 1)
        )  # first transition matrix is always identity to get i.c., last step also included
        jump_inds = jump_inds + 1
        
        As = As.at[:, trans_inds].set(As_).at[:, jump_inds].set(Ajs)
        Qs = Qs.at[:, trans_inds].set(Qs_).at[:, jump_inds].set(Qjs)
        
        return H, Pinf, As, Qs

    ### posterior ###
    def evaluate_conditional_posterior(
        self, t_eval, j_path, mean_only, compute_KL, jitter
    ):
        """
        Compute posterior conditioned on HMM path

        :param jnp.ndarray j_path: (num_samps, path_locs)
        """
        all_locs, trans_inds, jump_inds, obs_inds, unique_dlocs = self.get_site_locs()

        H, Pinf, As, Qs = self.get_conditional_LDS(
            unique_dlocs, trans_inds, jump_inds, j_path
        )

        all_obs, all_Lcovs = self.get_site_params(obs_inds, len(all_locs))

        # interpolation
        ind_eval, dt_fwd, dt_bwd = interpolation_times(t_eval, all_locs)
        dt_fwd_bwd = jnp.concatenate([dt_fwd, dt_bwd])  # (2*num_evals,)
        A_fwd_bwd = vmap(self.kernel.state_transition)(
            dt_fwd_bwd
        )  # vmap over num_evals
        Q_fwd_bwd = vmap(LTI_process_noise, (0, None), 0)(A_fwd_bwd, Pinf)
        
        post_means, post_covs, KL = vLGSSM(
            H,
            Pinf,
            Pinf,
            As,
            Qs,
            all_obs,
            all_Lcovs,
            ind_eval,
            A_fwd_bwd,
            Q_fwd_bwd,
            mean_only,
            compute_KL,
            jitter,
        )

        return post_means[:, obs_inds], post_covs[:, obs_inds], KL

    def evaluate_posterior(
        self, prng_state, num_samps, t_eval, mean_only, compute_KL, jitter
    ):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (time,)
        :return:
            means of shape (time, out_dims, 1)
            covariances of shape (time, out_dims, out_dims)
        """
        # sample from HMMs
        j_path, KL_j = self.switch_HMM.sample_posterior(
            prng_state, num_samps, compute_KL
        )

        post_means, post_covs, KL_x = self.evaluate_conditional_posterior(
            t_eval, j_path, mean_only, compute_KL, jitter
        )

        KL = KL_x + KL_j
        return post_means, post_covs, KL

    ### sample ###
    def sample_conditional_prior(self, prng_states, t_eval, j_path, jitter):
        """
        Sample from the model prior f~N(0,K) via simulation of the LGSSM

        :param int num_samps: number of samples to draw
        :param jnp.ndarray t_eval: the input locations at which to sample (out_dims, locs,)
        :return:
            f_sample: the prior samples (num_samps, time, out_dims, 1)
        """
        num_samps = prng_states.shape[0]
        all_locs, trans_inds, jump_inds, obs_inds, unique_locs = obs_switch_locs(
            t_eval, self.switch_locs[(t_eval[0] < self.switch_locs) & (t_eval[-1] > self.switch_locs)]
        )
        
        unique_dlocs = jnp.diff(unique_locs)  # (eval_locs-1,)
        if jnp.all(jnp.isclose(unique_dlocs, unique_dlocs[0])):  # grid
            unique_dlocs = unique_dlocs[:1]

        H, Pinf, As, Qs = self.get_conditional_LDS(
            unique_dlocs, trans_inds, jump_inds, j_path
        )
        m0 = jnp.zeros((Pinf.shape[-1], 1))

        samples = vsample(H, m0, Pinf, As, Qs, prng_states, 1, jitter)
        x_samples = samples[:, obs_inds, 0, ...]
        return x_samples

    def sample_conditional_posterior(
        self,
        prng_states,
        t_eval,
        j_path,
        jitter,
        compute_KL,
    ):
        prng_state, prng_states = prng_states[0], prng_states[1:]
        all_locs, trans_inds, jump_inds, obs_inds, unique_locs = obs_switch_locs(
            self.site_locs, self.switch_locs
        )
        all_obs, all_Lcovs = self.get_site_params(obs_inds, len(all_locs))

        # compute linear dynamical system
        unique_dlocs = jnp.diff(unique_locs)  # (eval_locs-1,)
        if jnp.all(jnp.isclose(unique_dlocs, unique_dlocs[0])):  # grid
            unique_dlocs = unique_dlocs[:1]
        H, Pinf, As, Qs = self.get_conditional_LDS(
            unique_dlocs, trans_inds, jump_inds, j_path
        )

        # evaluation locations
        t_all, site_ind, eval_ind = order_times(t_eval, all_locs)
        if t_eval is not None:
            num_evals = t_eval.shape[0]
            ind_eval, dt_fwd, dt_bwd = interpolation_times(t_eval, all_locs)
            dt_fwd_bwd = jnp.concatenate([dt_fwd, dt_bwd])  # (2*num_evals,)
            A_fwd_bwd = vmap(self.kernel.state_transition)(
                dt_fwd_bwd
            )  # vmap over num_evals
            Q_fwd_bwd = vmap(LTI_process_noise, (0, None), 0)(A_fwd_bwd, Pinf)

        else:
            ind_eval, A_fwd_bwd, Q_fwd_bwd = None, None, None

        # posterior mean
        post_means, _, KL_x = vLGSSM(
            H,
            Pinf,
            Pinf,
            As,
            Qs,
            all_obs,
            all_Lcovs,
            ind_eval,
            A_fwd_bwd,
            Q_fwd_bwd,
            True,
            compute_KL,
            jitter,
        )  # (tr, time, out_dims, 1)
        
        # sample prior at obs and eval locs
        prior_samps = self.sample_conditional_prior(prng_states, t_all, j_path, jitter)

        # noisy prior samples at eval locs
        prior_samps_t = prior_samps[:, site_ind]
        prior_samps_eval = prior_samps[:, eval_ind]

        prior_samps_noisy = prior_samps_t + all_Lcovs[None, ...] @ jr.normal(
            prng_state, shape=prior_samps_t.shape
        )  # (tr, time, out_dims, 1)

        # smooth noisy samples
        def smooth_prior_sample(prior_samp_i, As, Qs):
            smoothed_sample, _, _ = evaluate_LGSSM_posterior(
                H,
                Pinf,
                Pinf,
                As,
                Qs,
                prior_samp_i,
                all_Lcovs,
                ind_eval,
                A_fwd_bwd,
                Q_fwd_bwd,
                mean_only=True,
                compute_KL=False,
                jitter=jitter,
            )
            return smoothed_sample

        smoothed_samps = vmap(smooth_prior_sample)(
            prior_samps_noisy, As, Qs
        )
        return prior_samps_eval - smoothed_samps + post_means, KL_x

    def sample_prior(self, prng_state, num_samps, t_eval, jitter):
        """
        Sample from the model prior f~N(0,K) via simulation of the LGSSM

        :param int num_samps: number of samples to draw
        :param jnp.ndarray t_eval: the input locations at which to sample (out_dims, locs,)
        :return:
            f_sample: the prior samples (num_samps, time, out_dims, 1)
        """
        # sample from HMMs
        switches = self.switch_locs.shape[0]
        j_path = self.switch_HMM.sample_prior(prng_state, num_samps, switches)
        prng_states = jr.split(prng_state, num_samps)
        x_samples = self.sample_conditional_prior(prng_states, t_eval, j_path, jitter)
        return x_samples, j_path

    def sample_posterior(
        self,
        prng_state,
        num_samps,
        t_eval,
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

        :param jnp.ndarray seed: JAX prng state
        :param int num_samps: the number of samples to draw
        :param jnp.ndarray t_eval: the sampling input locations (eval_nums,), if None, sample at site locs
        :return:
            the posterior samples (eval_locs, num_samps, N, 1)
        """
        # sample from HMMs
        j_path, KL_j = self.switch_HMM.sample_posterior(
            prng_state, num_samps, compute_KL
        )
        prng_keys = jr.split(prng_state, num_samps + 1)
        x_samples, KL_x = self.sample_conditional_posterior(
            prng_keys, t_eval, j_path, jitter, compute_KL
        )
        KL = KL_x + KL_j
        return x_samples, j_path, KL


class SwitchingLTI(SSM):
    """
    In the switching formulation, every observation point is doubled to account
    for infinitesimal switches inbetween

    GP state can also switch in addition to jumps
    """

    switch_locs: jnp.ndarray
    state_HMM: DTMarkovChain
    switch_HMM: DTMarkovChain

    switch_transitions: Transition
    markov_kernel_group: GroupMarkovian

    fixed_grid_locs: bool

    def __init__(
        self,
        switch_locs,
        state_HMM,
        switch_HMM,
        markov_kernel_group,
        switch_transitions,
        site_locs,
        site_obs,
        site_Lcov,
        fixed_grid_locs=False,
    ):
        """
        :param jnp.ndarray switch_locs: locations of switches (K, K, switches)
        """
        assert switch_HMM.site_ll.shape[0] == switch_locs.shape[0]
        assert state_HMM.site_ll.shape[0] == switch_locs.shape[0] + 1
        assert state_HMM.array_type == switch_HMM.array_type
        assert state_HMM.array_type == markov_kernel_group.array_type
        assert state_HMM.array_type == switch_transitions.array_type
        super().__init__(
            site_locs, site_obs, site_Lcov, ArrayTypes_[state_HMM.array_type]
        )
        self.switch_locs = self._to_jax(switch_locs)
        self.state_HMM = state_HMM
        self.switch_HMM = switch_HMM

        self.switch_transitions = switch_transitions
        self.markov_kernel_group = markov_kernel_group

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

    ### site and switch locations ###
    def get_site_locs(self):
        """
        Both site obervation locs and switch locs contribute the site locations
        """
        locs, obs_inds, switch_inds, unique_locs = obs_switch_locs(
            self.site_locs, self.switch_locs
        )

        if fixed_grid_locs:
            locs = lax.stop_gradient(locs)  # (ts,)
            return locs, obs_inds, switch_inds, unique_locs[1:2] - unique_locs[0:1]
        else:
            return locs, obs_inds, switch_inds, jnp.diff(unique_locs)

    def get_site_params(self, obs_inds, ts):
        """
        Insert unobserved locations at switch edge locations
        
        :param int ts: total time points = len(obs_inds) + len(switch_inds)
        """
        sd = self.site_obs.shape[-2]

        site_obs = jnp.zeros((ts, sd, 1))
        site_Lcov = 1e6 * jnp.eye(sd)[None, ...].repeat(
            ts, axis=0
        )  # empty observations

        site_obs = site_obs.at[obs_inds].set(self.site_obs)
        site_Lcov = site_Lcov.at[obs_inds].set(self.site_Lcov)
        return site_obs, site_Lcov

    def get_conditional_LDS(self, dt, tsteps, obs_inds, switch_inds, s_path, j_path):
        """
        Insert switches inbetween observation sites
        Each observation site has a state
        Each switch applies from A_{s}(t/2) * A_{ss'} * A_{s'}(t/2) with empty
        observation at the * locations

        :param jnp.ndarray s_path: GP state trajectories (num_samps, switch_locs + 1), [0, s_dims)
        :param jnp.ndarray j_path: switch state trajectories (num_samps, switch_locs), [0, j_dims)
        """
        switches, observes, sd = (
            len(switch_inds),
            len(obs_inds),
            self.markov_kernel_group.state_dims,
        )
        trans_without_jumps = observes + switches // 2 - 1
        trans_with_jumps = observes + switches + 1  # include boundary transitions

        Ps = jnp.empty((trans_without_jumps, sd, sd))
        As = jnp.empty((trans_with_jumps, sd, sd))
        Qs = jnp.empty((trans_with_jumps, sd, sd))

        # expand s_path
        s_path = vmap(tile_between_inds(0, None, None), 0)(
            s_path, switch_inds, trans_without_jumps
        )  # (num_samps, ts)

        # use out_dims of kernel as states
        s_steps = jnp.arange(trans_without_jumps)
        H, Ps_, As_, Qs_ = self.markov_kernel_group.get_LDS(
            dt, trans_without_jumps
        )  # (states, ts, sd, sd)
        Ps_ = Ps_[s_path, s_steps]  # (ts, sd, sd)
        As_, Qs_ = (
            As_[s_path[1:-1], s_steps[:-2]],
            Qs_[s_path[1:-1], s_steps[:-2]],
        )  # (ts, sd, sd)

        # jump matrices
        Ajs, Qjs = self.transitions.state_transition()  # (states, sd, sd)
        Ajs, Qjs = Ajs[j_path], Qjs[j_path]  # (ts, sd, sd)

        # insert matrices
        trans_inds = jnp.concatenate((obs_inds, switch_inds[1::2]))
        trans_inds = jnp.delete(
            trans_inds, jnp.argmax(trans_inds)
        )  # remove last edge point
        jump_inds = switch_inds[::2]
        As = As.at[trans_inds + 1].set(As_).at[jump_inds + 1].set(Ajs)
        Qs = Qs.at[trans_inds + 1].set(Qs_).at[jump_inds + 1].set(Qjs)

        # convenience boundary transitions for kalman filter and smoother
        Id = jnp.eye(As.shape[-1])
        Zs = jnp.zeros_like(Id)
        As = As.at[jnp.array([0, -1])].set(Id)
        Qs = Qs.at[jnp.array([0, -1])].set(Zs)

        return H, Ps, As, Qs

    ### posterior ###
    def evaluate_conditional_posterior(self, t_eval, s_path, j_path):
        """
        Compute posterior conditioned on HMM path
        """
        site_locs, obs_inds, switch_inds, site_dlocs = self.get_site_locs()

        site_obs, site_Lcovs = self.get_site_params(obs_inds, len(all_locs))
        H, Ps, As, Qs = self.get_conditional_LDS(
            site_dlocs, tsteps, obs_inds, switch_inds, s_path, j_path
        )

        # interpolation
        ind_eval, dt_fwd, dt_bwd = interpolation_times(t_eval, site_locs)
        dt_fwd_bwd = jnp.concatenate([dt_fwd, dt_bwd])  # (2*num_evals,)
        P_fwd_bwd = jnp.tile(Ps[ind_eval], 2)
        A_fwd_bwd = vmap(self.markov_kernel_group.state_transition)(
            dt_fwd_bwd
        )  # vmap over num_evals
        A_fwd_bwd = A_fwd_bwd[jnp.tile(s_path, 2)]  # select states
        Q_fwd_bwd = vmap(LTI_process_noise, (0, 0), 0)(A_fwd_bwd, P_fwd_bwd)

        post_means, post_covs, KL = evaluate_LGSSM_posterior(
            H,
            Ps[0],
            Ps[-1],
            As,
            Qs,
            site_obs,
            site_Lcov,
            ind_eval,
            A_fwd_bwd,
            Q_fwd_bwd,
            mean_only,
            compute_KL,
            jitter,
        )

        return post_means, post_covs, KL

    def evaluate_posterior(self, prng_state, t_eval, mean_only, compute_KL, jitter):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (time,)
        :return:
            means of shape (time, out_dims, 1)
            covariances of shape (time, out_dims, out_dims)
        """
        # sample from HMMs
        s_path, KL_s = self.state_HMM.sample_posterior(
            prng_state, num_samps, compute_KL
        )
        prng_state, _ = jr.split(prng_state)
        j_path, KL_j = self.switch_HMM.sample_posterior(
            prng_state, num_samps, compute_KL
        )

        post_means, post_covs, KL_x = self.evaluate_conditional_posterior(
            t_eval, s_path, j_path
        )

        KL = KL_x + KL_s + KL_j
        return post_means, post_covs, KL

    ### sample ###
    def sample_prior(self, prng_state, num_samps, t_eval, jitter):
        """
        Sample from the model prior f~N(0,K) via simulation of the LGSSM

        :param int num_samps: number of samples to draw
        :param jnp.ndarray t_eval: the input locations at which to sample (out_dims, locs,)
        :return:
            f_sample: the prior samples (num_samps, time, out_dims, 1)
        """
        locs, obs_inds, switch_inds, unique_locs = obs_switch_locs(
            t_eval, self.switch_locs
        )

        ts = unique_locs.shape[0]
        switches = self.switch_locs.shape[0]
        dt = jnp.diff(unique_locs)  # (eval_locs-1,)
        if jnp.all(jnp.isclose(dt, dt[0])):  # grid
            dt = dt[:1]

        # sample from HMMs
        prng_states = jr.split(prng_state, 3)

        s_path = self.state_HMM.sample_prior(prng_states[0], num_samps, ts + 1)
        j_path = self.switch_HMM.sample_prior(prng_states[1], num_samps, switches)

        H, Ps, As, Qs = self.get_conditional_LDS(
            dt, ts, obs_inds, switch_inds, s_path, j_path
        )

        samples = sample_LGSSM(H, minf, Pinf, As, Qs, prng_states[2], num_samps, jitter)
        x_samples = samples.transpose(1, 0, 2, 3)

        return x_samples, s_path, j_path

    def sample_posterior(
        self,
        prng_state,
        num_samps,
        t_eval,
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

        :param jnp.ndarray seed: JAX prng state
        :param int num_samps: the number of samples to draw
        :param jnp.ndarray t_eval: the sampling input locations (eval_nums,), if None, sample at site locs
        :return:
            the posterior samples (eval_locs, num_samps, N, 1)
        """
        # sample from HMMs
        s_path, KL_s = self.state_HMM.sample_posterior(prng_state)
        prng_state, _ = jr.split(prng_state)
        j_path, KL_j = self.switch_HMM.sample_posterior(prng_state)

        site_locs, site_dlocs = self.get_site_locs()

        # evaluation locations
        t_all, site_ind, eval_ind = order_times(t_eval, site_locs)
        if t_eval is not None:
            num_evals = t_eval.shape[0]
            ind_eval, dt_fwd, dt_bwd = interpolation_times(t_eval, site_locs)

            stack_dt = jnp.concatenate([dt_fwd, dt_bwd])  # (2*num_evals,)
            stack_A = vmap(self.markov_kernel_group.state_transition)(
                stack_dt
            )  # vmap over num_evals
            A_fwd, A_bwd = stack_A[:num_evals], stack_A[-num_evals:]

        else:
            ind_eval, A_fwd, A_bwd = None, None, None

        # compute linear dynamical system
        H, Ps, As, Qs = self.markov_kernel_group.get_LDS(
            site_dlocs[None, :], site_locs.shape[0]
        )

        # sample prior at obs and eval locs
        prng_keys = jr.split(prng_state, 2)
        prior_samps = self.sample_prior(
            prng_keys[0], num_samps, t_all, jitter
        )  # (time, num_samps, out_dims, 1)

        # posterior mean
        post_means, _, KL_x = evaluate_LGSSM_posterior(
            H,
            Ps[0],
            Ps[-1],
            As,
            Qs,
            site_obs,
            site_Lcov,
            ind_eval,
            A_fwd_bwd,
            Q_fwd_bwd,
            mean_only=True,
            compute_KL=compute_KL,
            jitter=jitter,
        )  # (time, out_dims, 1)

        # noisy prior samples at eval locs
        prior_samps_t = prior_samps[:, site_ind]
        prior_samps_eval = prior_samps[:, eval_ind]

        prior_samps_noisy = prior_samps_t + self.site_Lcov[None, ...] @ jr.normal(
            prng_keys[1], shape=prior_samps_t.shape
        )  # (time, tr, out_dims, 1)

        # smooth noisy samples
        def smooth_prior_sample(prior_samp_i):
            smoothed_sample, _, _ = evaluate_LGSSM_posterior(
                H,
                Ps[0],
                Ps[-1],
                As,
                Qs,
                prior_samp_i,
                site_Lcov,
                ind_eval,
                A_fwd_bwd,
                Q_fwd_bwd,
                mean_only=True,
                compute_KL=False,
                jitter=jitter,
            )
            return smoothed_sample

        smoothed_samps = vmap(smooth_prior_sample, 0, 0)(prior_samps_noisy)

        KL = KL_x + KL_s + KL_j

        # Matheron's rule pathwise samplig
        return (
            prior_samps_eval - smoothed_samps + post_means[None, ...],
            s_path,
            j_path,
            KL_ss,
        )

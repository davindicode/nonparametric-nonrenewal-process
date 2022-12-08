import math
from functools import partial

from .base import module

import jax.numpy as jnp
import jax.random as jr
from jax import grad, jacrev, jit, lax, tree_map, vmap
from jax.numpy.linalg import cholesky

from jax.scipy.linalg import cho_solve, solve_triangular

from ..utils.linalg import solve

_log_twopi = math.log(2 * math.pi)


# linear algebra operations
def kalman_filter(
    obs_means, obs_Lcovs, As, Qs, H, minf, Pinf, return_predict=False, compute_logZ=True
):
    """
    Run the Kalman filter to get p(fₙ|y₁,...,yₙ).

    If store is True then we compute and return the intermediate filtering distributions
    p(fₙ|y₁,...,yₙ) and sites sₙ(fₙ), otherwise we do not store the intermediates and simply
    return the energy / negative log-marginal likelihood, -log p(y).

    :param params: the model parameters, i.e the hyperparameters of the prior & likelihood
    :param store: flag to notify whether to store the intermediates
    :return:
    """
    D = obs_means.shape[1]

    def step(carry, inputs):
        pseudo_mean, pseudo_Lcov, A, Q = inputs
        f_m, f_P, lZ = carry

        # mₙ⁻ = Aₙ mₙ₋₁
        # Pₙ⁻ = Aₙ Pₙ₋₁ Aₙ' + Qₙ, where Qₙ = Pinf - Aₙ Pinf Aₙ'
        m_ = A @ f_m
        P_ = A @ f_P @ A.T + Q

        predict_mean = H @ m_
        HP = H @ P_
        predict_cov = HP @ H.T

        S = predict_cov + pseudo_Lcov @ pseudo_Lcov.T
        K = solve(S, HP).T
        eta = pseudo_mean - predict_mean

        f_m = m_ + K @ eta
        f_P = P_ - K @ HP

        if compute_logZ:
            L = cholesky(S)  # + jitter*I)
            n = solve_triangular(L, eta, lower=True)  # (x_dims, 1)
            lZ -= jnp.log(jnp.diag(L)).sum() + 0.5 * (jnp.sum(n * n)) + D * _log_twopi

        carry = (f_m, f_P, lZ)
        out = (m_, P_) if return_predict else (f_m, f_P)
        return carry, out

    init = (minf, Pinf, 0.0)
    xs = (obs_means, obs_Lcovs, As, Qs)

    carry, (filter_means, filter_covs) = lax.scan(step, init, xs)
    log_Z = carry[2]
    return log_Z, filter_means, filter_covs


def rauch_tung_striebel_smoother(
    filter_means, filter_covs, As, Qs, H, full_state=False
):
    """
    Run the RTS smoother to get p(fₙ|y₁,...,y_N)

    :param m_filtered: the intermediate distribution means computed during filtering [N, state_dim, 1]
    :param P_filtered: the intermediate distribution covariances computed during filtering [N, state_dim, state_dim]
    :param store: a flag determining whether to store and return state mean and covariance
    :param return_full: a flag determining whether to return the full state distribution or just the function(s)
    :return:
        var_exp: the sum of the variational expectations [scalar]
        smoothed_mean: the posterior marginal means [N, obs_dim]
        smoothed_var: the posterior marginal variances [N, obs_dim]
    """

    def step(carry, inputs):
        f_m, f_P, A, Q = inputs
        s_m, s_P = carry

        predict_m = A @ f_m
        A_f_P = A @ f_P
        predict_P = A_f_P @ A.T + Q

        G = solve(predict_P, A_f_P).T  # G = F * A' * P^{-1} = (P^{-1} * A * F)'

        s_m = f_m + G @ (s_m - predict_m)
        s_P = f_P + G @ (s_P - predict_P) @ G.T

        carry = (s_m, s_P)
        out = (s_m, s_P, G) if full_state else (H @ s_m, H @ s_P @ H.T, G)
        return carry, out

    init = (filter_means[-1], filter_covs[-1])
    xs = (filter_means, filter_covs, As, Qs)

    _, (smoother_means, smoother_covs, Gs) = lax.scan(step, init, xs, reverse=True)
    return smoother_means, smoother_covs, Gs


def process_noise_covariance(A, Pinf):
    Q = Pinf - A @ Pinf @ A.T
    return Q


def pseudo_log_likelihood(post_mean, post_cov, site_obs, site_Lcov):
    """ """
    D = site_obs.shape[1]
    m = site_obs - post_mean  # (time, N, 1)
    tr_term = jnp.trace(
        cho_solve((site_Lcov, True), m * m.transpose(0, 2, 1) + post_cov).sum(0)
    )
    site_log_lik = -jnp.log(vmap(jnp.diag)(site_Lcov)).sum() - 0.5 * (
        tr_term + D * _log_twopi
    )
    # diagonal of site_Lcov is thresholded in site_update() and constraints() to be >= 1e-3
    return site_log_lik


def compute_conditional_statistics(
    kernel, params, Pinf, t_eval, t, ind, mean_only, jitter
):
    """
    Predicts marginal states at new time points. (new time points should be sorted)
    Calculates the conditional density:
             p(xₙ|u₋, u₊) = 𝓝(Pₙ @ [u₋, u₊], Tₙ)
    :param x_test: time points to generate observations for [N]
    :param x: inducing state input locations [M]
    :param kernel: prior object providing access to state transition functions
    :param ind: an array containing the index of the inducing state to the left of every input [N]
    :return: parameters for the conditional mean and covariance
            P: [N, D, 2*D]
            T: [N, D, D]
    """
    dt_fwd = t_eval - t[ind]
    dt_back = t[ind + 1] - t_eval
    A_fwd = kernel.state_transition(dt_fwd, params)
    A_back = kernel.state_transition(dt_back, params)

    Q_fwd = Pinf - A_fwd @ Pinf @ A_fwd.T
    Q_back = Pinf - A_back @ Pinf @ A_back.T
    A_back_Q_fwd = A_back @ Q_fwd
    Q_mp = Q_back + A_back @ A_back_Q_fwd.T

    eps = jitter * jnp.eye(Q_mp.shape[0])
    chol_Q_mp = cholesky(Q_mp + eps)
    Q_mp_inv_A_back = cho_solve((chol_Q_mp, True), A_back)  # V = Q₋₊⁻¹ Aₜ₊

    # W = Q₋ₜAₜ₊ᵀQ₋₊⁻¹
    W = Q_fwd @ Q_mp_inv_A_back.T
    P = jnp.concatenate([A_fwd - W @ A_back @ A_fwd, W], axis=-1)

    if mean_only:
        return P
    else:
        # The conditional_covariance T = Q₋ₜ - Q₋ₜAₜ₊ᵀQ₋₊⁻¹Aₜ₊Q₋ₜ == Q₋ₜ - Q₋ₜᵀAₜ₊ᵀL⁻ᵀL⁻¹Aₜ₊Q₋ₜ
        T = Q_fwd - A_back_Q_fwd.T @ Q_mp_inv_A_back @ Q_fwd
        return P, T


def predict_from_state(
    t_eval,
    ind,
    t,
    post_mean,
    post_cov,
    gain,
    kernel,
    kern_params,
    Pinf,
    mean_only,
    jitter,
):
    """
    predict the state distribution at time t by projecting from the neighbouring inducing states
    """
    # joint posterior (i.e. smoothed) mean and covariance of the states [u_, u+] at time t:
    mean_joint = jnp.block([[post_mean[ind]], [post_mean[ind + 1]]])

    if mean_only is False:
        P, T = compute_conditional_statistics(
            kernel, kern_params, Pinf, t_eval, t, ind, mean_only, jitter
        )
        cross_cov = gain[ind] @ post_cov[ind + 1]
        cov_joint = jnp.block(
            [[post_cov[ind], cross_cov], [cross_cov.T, post_cov[ind + 1]]]
        )
        return P @ mean_joint, P @ cov_joint @ P.T + T

    else:
        P = compute_conditional_statistics(
            kernel, kern_params, Pinf, t_eval, t, ind, mean_only, jitter
        )
        return P @ mean_joint


def get_evaldata(t_eval, timedata):
    """ """
    t = timedata[0]
    tsteps = t.shape[0]  # assume timedata is ordered and unique

    t_all = jnp.concatenate([t, t_eval], axis=0)
    t_all, input_ind = jnp.unique(t_all, return_inverse=True)

    sort_ind = jnp.argsort(t_all, axis=0)
    t_all = t_all[sort_ind]
    all_timedata = (t_all, jnp.diff(t_all))
    t_ind, eval_ind = sort_ind[input_ind[:tsteps]], sort_ind[input_ind[tsteps:]]
    return all_timedata, t_ind, eval_ind


### classes ###
class StateSpace(module):
    """
    The state space model class
    """
    x_dims: int
    diagonal_site: bool
    kernel: module
        
    site_obs: jnp.ndarray
    site_Lcov: jnp.ndarray

    def __init__(self, x_dims, kernel, site_obs, site_Lcov, diagonal_site):
        """
        :param dict hyp: (hyper)parameters of the state space model
        :param dict var_params: variational parameters
        """
        self.x_dims = x_dims
        self.diagonal_site = diagonal_site
        self.kernel = kernel
        
        self.site_obs = site_obs
        self.site_Lcov = site_Lcov

    def apply_constraints(self):
        """
        PSD constraint
        """
        model = jax.tree_map(lambda p: p, self)  # copy
        
        def update(W_rec):
            Lcov = var_params["site_Lcov"]
            epdfunc = lambda x: enforce_positive_diagonal(x, lower_lim=1e-2)
            Lcov = vmap(epdfunc)(jnp.tril(Lcov))
            var_params["site_Lcov"] = jnp.triu(Lcov) if self.diagonal_site else Lcov
            return params, var_params

        model = eqx.tree_at(
            lambda tree: tree.site_Lcov,
            model,
            replace_fn=update,
        )
        
        kernel = self.kernel.apply_constraints(self.kernel)
        model = eqx.tree_at(
            lambda tree: tree.kernel,
            model,
            replace_fn=lambda _: kernel,
        )

        return model
    
    def compute_intermediates(self, hyp, dt=None):
        """
        Compute intermediate reused values that only depend on hyperparameters
        """
        return {}  # empty container

    ### LDS ###
    def state_transition(self, dt, hyp=None):
        hyp = self.kernel.hyp if hyp is None else hyp
        return self.kernel.state_transition(dt, hyp)

    def state_output(self, hyp=None):
        hyp = self.kernel.hyp if hyp is None else hyp
        return self.kernel.state_output(hyp)


class FullLDS(StateSpace):
    """
    Full state space LDS
    """

    def __init__(self, kernel, var_params=None, diagonal_site=True):
        """
        :param dict hyp: (hyper)parameters of the state space model
        :param dict var_params: if None, Kalman filter will initialize
        """
        super().__init__(kernel.out_dims, kernel, var_params, diagonal_site)

    def get_LDS_matrices(self, timedata, Pinf):
        t, dt = timedata
        Id = jnp.eye(Pinf.shape[0])
        Zs = jnp.zeros_like(Pinf)

        if dt.shape[0] == 1:
            A = self.state_transition(dt[0], params)
            Q = process_noise_covariance(A, Pinf)
            As = jnp.stack([Id] + [A] * (t.shape[0] - 1) + [Id], axis=0)
            Qs = jnp.stack([Zs] + [Q] * (t.shape[0] - 1) + [Zs], axis=0)
        else:
            As = vmap(self.state_transition, (0, None), 0)(dt, params)
            Qs = vmap(process_noise_covariance, (0, None), 0)(As, Pinf)
            As = jnp.concatenate((Id[None, ...], As, Id[None, ...]), axis=0)
            Qs = jnp.concatenate((Zs[None, ...], Qs, Zs[None, ...]), axis=0)

        return As, Qs

    ### posterior ###
    @partial(jit, static_argnums=(0, 5, 6))
    def evaluate_posterior(
        self, t_eval, timedata, mean_only, compute_KL, jitter
    ):
        """
        predict at test locations X, which may includes training points
        (which are essentially fixed inducing points)

        :param jnp.ndarray t_eval: evaluation times of shape (locs,)
        :return:
            means of shape (time, out, 1)
            covariances of shape (time, out, out)
        """
        H, minf, Pinf = self.kernel.state_output(hyp=params["kernel"])
        minf = minf[:, None]
        site_obs, site_Lcov = var_params["site_obs"], var_params["site_Lcov"]

        # compute linear dynamical system
        As, Qs = self.get_LDS_matrices(params["kernel"], timedata, Pinf)

        # filtering then smoothing
        logZ, filter_means, filter_covs = kalman_filter(
            site_obs,
            site_Lcov,
            As[:-1],
            Qs[:-1],
            H,
            minf,
            Pinf,
            return_predict=False,
            compute_logZ=compute_KL,
        )
        smoother_means, smoother_covs, gains = rauch_tung_striebel_smoother(
            filter_means, filter_covs, As[1:], Qs[1:], H, full_state=True
        )

        if t_eval is not None:  # predict the state distribution at the test time steps
            # add dummy states at either edge
            inf = 1e10 * jnp.ones(1)
            t_aug = jnp.concatenate([-inf, timedata[0], inf], axis=0)
            mean_aug = jnp.concatenate(
                [minf[None, :], smoother_means, minf[None, :]], axis=0
            )
            cov_aug = jnp.concatenate(
                [Pinf[None, ...], smoother_covs, Pinf[None, ...]], axis=0
            )
            gain_aug = jnp.concatenate([jnp.zeros_like(gains[:1, ...]), gains], axis=0)

            in_shape = tree_map(lambda x: None, params["kernel"])
            predict_from_state_vmap = vmap(
                predict_from_state,
                (0, 0, None, None, None, None, None, in_shape, None, None, None),
                0 if mean_only else (0, 0),
            )

            ind_eval = jnp.searchsorted(t_aug, t_eval) - 1
            if mean_only:
                eval_mean = predict_from_state_vmap(
                    t_eval,
                    ind_eval,
                    t_aug,
                    mean_aug,
                    cov_aug,
                    gain_aug,
                    self.kernel,
                    params["kernel"],
                    Pinf,
                    True,
                    jitter,
                )
                post_means, post_covs, KL = H @ eval_mean, None, 0.0

            else:
                eval_mean, eval_cov = predict_from_state_vmap(
                    t_eval,
                    ind_eval,
                    t_aug,
                    mean_aug,
                    cov_aug,
                    gain_aug,
                    self.kernel,
                    params["kernel"],
                    Pinf,
                    False,
                    jitter,
                )
                post_means, post_covs, KL = H @ eval_mean, H @ eval_cov @ H.T, 0.0

        else:
            post_means = H @ smoother_means
            if compute_KL:  # compute using pseudo likelihood
                post_covs = H @ smoother_covs @ H.T
                site_log_lik = pseudo_log_likelihood(
                    post_means, post_covs, site_obs, site_Lcov
                )
                KL = site_log_lik - logZ

            else:
                post_covs = None if mean_only else H @ smoother_covs @ H.T
                KL = 0.0

        return post_means, post_covs, KL

    ### sample ###
    @partial(jit, static_argnums=(0, 3))
    def sample_prior(self, params, prng_state, num_samps, timedata, jitter):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param t: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            f_sample: the prior samples [S, N_samp]
        """
        eps_I = jitter * jnp.eye(self.kernel.state_dims)
        H, minf, Pinf = self.state_output(hyp=params["kernel"])

        # transition and noise process matrices
        tsteps = timedata[0].shape[0]
        As, Qs = self.get_LDS_matrices(params["kernel"], timedata, Pinf)

        prng_states = jr.split(prng_state, num_samps)  # (num_samps, 2)

        def step(carry, inputs):
            m = carry
            A, Q, prng_state = inputs
            L = cholesky(Q + eps_I)  # can be a bit unstable, lower=True

            q_samp = L @ jr.normal(prng_state, shape=(self.kernel.state_dims, 1))
            m = A @ m + q_samp
            f = H @ m
            return m, f

        def sample_i(prng_state):
            m0 = cholesky(Pinf) @ jr.normal(
                prng_state, shape=(self.kernel.state_dims, 1)
            )
            prng_keys = jr.split(prng_state, tsteps)
            _, f_sample = lax.scan(step, init=m0, xs=(As[:-1], Qs[:-1], prng_keys))
            return f_sample

        f_samples = vmap(sample_i, 0, 1)(prng_states)
        return f_samples  # (time, tr, state_dims, 1)

    @partial(jit, static_argnums=(0, 4, 8))
    def sample_posterior(
        self,
        params,
        var_params,
        prng_state,
        num_samps,
        timedata,
        evaldata,
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
        site_obs, site_Lcov = var_params["site_obs"], var_params["site_Lcov"]

        if evaldata is None:
            all_timedata = timedata
            t_ind = jnp.arange(timedata[0].shape[0])
            eval_ind = t_ind
        else:
            all_timedata, t_ind, eval_ind = evaldata
        t_eval = all_timedata[0][eval_ind]

        prng_keys = jr.split(prng_state, 2)

        prior_samps = self.sample_prior(
            params, prng_keys[0], num_samps, all_timedata, jitter
        )
        post_means, _, KL_ss = self.evaluate_posterior(
            t_eval,
            timedata,
            params,
            var_params,
            mean_only=True,
            compute_KL=compute_KL,
            jitter=jitter,
        )  # (time, N, 1)

        prior_samps_t, prior_samps_eval = (
            prior_samps[t_ind, ...],
            prior_samps[eval_ind, ...],
        )
        prior_samps_noisy = prior_samps_t + site_Lcov[:, None, ...] @ jr.normal(
            prng_keys[1], shape=prior_samps_t.shape
        )  # (time, tr, N, 1)

        def smooth_prior_sample(prior_samp_i):
            var_params = {"site_obs": prior_samp_i, "site_Lcov": site_Lcov}
            smoothed_sample, _, _ = self.evaluate_posterior(
                t_eval,
                timedata,
                params,
                var_params,
                mean_only=True,
                compute_KL=False,
                jitter=jitter,
            )
            return smoothed_sample

        smoothed_samps = vmap(smooth_prior_sample, 1, 1)(prior_samps_noisy)
        # Matheron's rule pathwise samplig
        return prior_samps_eval - smoothed_samps + post_means[:, None, ...], KL_ss


class UncoupledLDS(StateSpace):
    """
    Uncoupled LDS (block diagonal state space)
    """

    def __init__(self, x_dims, kernel, var_params=None, diagonal_site=True):
        """
        :param dict hyp: (hyper)parameters of the state space model
        :param dict var_params: if None, Kalman filter will initialize
        """
        super().__init__(x_dims, kernel, var_params, diagonal_site)
        self.hyp = tree_map(lambda h: h[None, ...].repeat(x_dims, axis=0), kernel.hyp)
        # if len(H.shape) == 3: # independent latent dimensions
        # vmap over kernel and kalman filter to get uncoupled dynamics

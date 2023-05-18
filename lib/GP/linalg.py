import math

import jax.numpy as jnp
from jax import lax, tree_map, vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import block_diag, cho_solve, solve_triangular

from ..utils.jax import safe_log

from ..utils.linalg import solve_PSD

_log_twopi = math.log(2 * math.pi)


# LGSSM
def kalman_filter(
    obs_means, obs_Lcovs, As, Qs, H, m0, P0, return_predict=False, compute_logZ=True
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
        obs_mean, obs_Lcov, A, Q = inputs
        f_m, f_P, lZ = carry

        m_ = A @ f_m
        P_ = A @ f_P @ A.T + Q

        predict_mean = H @ m_
        HP = H @ P_
        predict_cov = HP @ H.T

        S = predict_cov + obs_Lcov @ obs_Lcov.T
        K = solve_PSD(S, HP).T
        eta = obs_mean - predict_mean

        f_m = m_ + K @ eta
        f_P = P_ - K @ HP

        if compute_logZ:
            L = cholesky(S)  # + jitter*I)
            n = solve_triangular(L, eta, lower=True)  # (state_dims, 1)
            lZ -= jnp.log(jnp.diag(L)).sum() + 0.5 * (jnp.sum(n * n)) + D * _log_twopi

        carry = (f_m, f_P, lZ)
        out = (m_, P_) if return_predict else (f_m, f_P)
        return carry, out

    init = (m0, P0, 0.0)
    xs = (obs_means, obs_Lcovs, As, Qs)

    carry, (filter_means, filter_covs) = lax.scan(step, init, xs)
    log_Z = carry[2]
    return log_Z, filter_means, filter_covs


def rauch_tung_striebel_smoother(
    filter_means, filter_covs, As, Qs, H, return_gains=True, full_state=False
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

        G = solve_PSD(predict_P, A_f_P).T  # G = F * A' * P^{-1} = (P^{-1} * A * F)'

        s_m = f_m + G @ (s_m - predict_m)
        s_P = f_P + G @ (s_P - predict_P) @ G.T

        carry = (s_m, s_P)
        out = (s_m, s_P) if full_state else (H @ s_m, H @ s_P @ H.T)
        out += (G,) if return_gains else (None,)
        return carry, out

    init = (filter_means[-1], filter_covs[-1])
    xs = (filter_means, filter_covs, As, Qs)

    _, (smoother_means, smoother_covs, Gs) = lax.scan(step, init, xs, reverse=True)
    return smoother_means, smoother_covs, Gs


def LTI_process_noise(A, Pinf):
    # Qₙ = Pinf - Aₙ Pinf Aₙ'
    Q = Pinf - A @ Pinf @ A.T
    return Q


def id_kronecker(dims, A):
    return jnp.kron(jnp.eye(dims), A)


def bdiag(M):
    """
    :param jnp.ndarray M: input matrices of shape (out_dims, sd, sd)
    """
    return block_diag(*M)  # (out_dims*sd, out_dims*sd)


def predict_between_sites(
    ind,
    A_fwd,
    A_bwd,
    Q_fwd,
    Q_bwd,
    post_mean,
    post_cov,
    gain,
    mean_only,
    jitter,
):
    """
    predict the state distribution at time t by projecting from the neighbouring inducing states

    Predicts marginal states at new time points. (new time points should be sorted)
    Calculates the conditional density:
             p(xₙ|u₋, u₊) = 𝓝(Pₙ @ [u₋, u₊], Tₙ)

    :param ind: time points indices to evaluate observations for (out_dims,)
    :param A_fwd: forward transitions from inducing states (out_dims, sd, sd)
    :param A_bwd: backward transitions from inducing states (out_dims, sd, sd)
    :return: parameters for the conditional mean and covariance
            P: [N, D, 2*D]
            T: [N, D, D]
    """
    A_bwd_Q_fwd = A_bwd @ Q_fwd
    Q_mp = Q_bwd + A_bwd @ A_bwd_Q_fwd.T

    eps = jitter * jnp.eye(Q_mp.shape[0])
    chol_Q_mp = cholesky(Q_mp + eps)
    Q_mp_inv_A_bwd = cho_solve((chol_Q_mp, True), A_bwd)  # V = Q₋₊⁻¹ Aₜ₊

    # W = Q₋ₜ Aₜ₊ᵀ Q₋₊⁻¹
    W = Q_fwd @ Q_mp_inv_A_bwd.T

    P = jnp.concatenate([A_fwd - W @ A_bwd @ A_fwd, W], axis=-1)

    # joint posterior mean of the states [u_, u+] at time t
    mean_joint = jnp.block([[post_mean[ind]], [post_mean[ind + 1]]])

    if mean_only:
        return P @ mean_joint, None

    else:
        # conditional covariance T = Q₋ₜ - Q₋ₜ Aₜ₊ᵀ Q₋₊⁻¹ Aₜ₊ Q₋ₜ = Q₋ₜ - Q₋ₜᵀ Aₜ₊ᵀ L⁻ᵀ L⁻¹ Aₜ₊ Q₋ₜ
        T = Q_fwd - A_bwd_Q_fwd.T @ Q_mp_inv_A_bwd @ Q_fwd
        cross_cov = gain[ind] @ post_cov[ind + 1]

        # joint posterior covariance of the states [u_, u+] at time t
        cov_joint = jnp.block(
            [[post_cov[ind], cross_cov], [cross_cov.T, post_cov[ind + 1]]]
        )
        return P @ mean_joint, P @ cov_joint @ P.T + T


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


def fixed_interval_smoothing(
    H, m0, P0, As, Qs, site_obs, site_Lcov, compute_logZ, return_gains
):
    """
    filtering then smoothing
    """
    logZ, filter_means, filter_covs = kalman_filter(
        site_obs,
        site_Lcov,
        As[:-1],
        Qs[:-1],
        H,
        m0,
        P0,
        return_predict=False,
        compute_logZ=compute_logZ,
    )
    smoother_means, smoother_covs, gains = rauch_tung_striebel_smoother(
        filter_means, filter_covs, As[1:], Qs[1:], H, return_gains, full_state=True
    )

    return smoother_means, smoother_covs, gains, logZ


predict_between_sites_vmap = vmap(
    predict_between_sites,
    (0, 0, 0, 0, 0, None, None, None, None, None),
    (0, 0),
)  # vmap over eval_nums


def evaluate_LGSSM_posterior(
    H,
    P_init,
    P_end,
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
):
    """
    predict at test locations X, which may includes training points
    (which are essentially fixed inducing points)

    :param jnp.ndarray P_init: stationary covariance at start (sd, sd)
    :param jnp.ndarray P_end: stationary covariance at end (sd, sd)
    :param jnp.ndarray As: transition matrices of shape (locs, sd, sd)
    :return:
        means of shape (time, out, 1)
        covariances of shape (time, out, out)
    """
    minf = jnp.zeros((As.shape[-1], 1))
    interpolate = ind_eval is not None

    smoother_means, smoother_covs, gains, logZ = fixed_interval_smoothing(
        H, minf, P_init, As, Qs, site_obs, site_Lcov, compute_KL, interpolate
    )

    if compute_KL:  # compute using pseudo likelihood
        post_means = H @ smoother_means
        post_covs = H @ smoother_covs @ H.T
    elif interpolate is False:  # evaluate at observed locs
        post_means = H @ smoother_means
        post_covs = None if mean_only else H @ smoother_covs @ H.T

    if compute_KL:
        site_log_lik = pseudo_log_likelihood(post_means, post_covs, site_obs, site_Lcov)
        KL = site_log_lik - logZ

    else:
        KL = 0.0

    if interpolate:  # predict the state distribution at the test time steps
        num_evals = len(ind_eval)
        A_fwd, A_bwd = A_fwd_bwd[:num_evals], A_fwd_bwd[-num_evals:]
        Q_fwd, Q_bwd = Q_fwd_bwd[:num_evals], Q_fwd_bwd[-num_evals:]

        # add dummy states at either edge
        mean_aug = jnp.concatenate(
            [minf[None, :], smoother_means, minf[None, :]], axis=0
        )
        cov_aug = jnp.concatenate(
            [P_init[None, ...], smoother_covs, P_end[None, ...]], axis=0
        )
        gain_aug = jnp.concatenate([jnp.zeros_like(gains[:1, ...]), gains], axis=0)

        eval_means, eval_covs = predict_between_sites_vmap(
            ind_eval,
            A_fwd,
            A_bwd,
            Q_fwd,
            Q_bwd,
            mean_aug,
            cov_aug,
            gain_aug,
            mean_only,
            jitter,
        )

        post_means = H @ eval_means
        post_covs = None if mean_only else H @ eval_covs @ H.T

    return post_means, post_covs, KL


# sparse
vdiag = vmap(jnp.diag)
vtrace = vmap(jnp.trace)


def mvn_conditional(Kxx, Kzx, Kzz, fz, mean_only, diag_cov, jitter):
    """
    Compute conditional of a MVN distribution p(f(x) | f(z))

    :param jnp.ndarray Kxx: (out_dims, eval_pts, eval_pts) or None
    :param jnp.ndarary Kzz: (out_dims, cond_pts, cond_pts)
    :param jnp.ndarray Kzx: (out_dims, cond_pts, eval_pts)
    :param fz jnp.ndarray: conditioned observations of shape (out_dims, cond_pts, 1)
    """
    eps_I = jitter * jnp.eye(Kzz.shape[-1])[None, ...]
    Lzz = cholesky(Kzz + eps_I)

    stacked = jnp.concatenate((fz, Kzx), axis=-1)
    Linv_stacked = solve_triangular(Lzz, stacked, lower=True)
    Linvf, LinvK = Linv_stacked[..., :1], Linv_stacked[..., 1:]
    W = LinvK.transpose(0, 2, 1)  # (out_dims, eval_pts, cond_pts)

    mean = W @ Linvf  # (out_dims, eval_pts, 1)
    if mean_only:
        return mean

    else:
        if diag_cov:
            cov = Kxx - (W**2).sum(-1, keepdims=True)
        else:
            cov = Kxx - W @ LinvK

        return mean, cov


def evaluate_qsparse_posterior(
    Kxx,
    Kzx,
    Kzz,
    u_mu,
    u_Lcov,
    whitened,
    mean_only,
    diag_cov,
    compute_KL,
    compute_aux,
    jitter,
):
    """
    :param jnp.ndarray Kxx: kernel (num_samps, out_dims, ts, ts) or None
    :param jnp.ndarray Kzx: kernel (out_dims, num_induc, num_samps, ts)
    :param jnp.ndarray Kzz: kernel (num_samps, out_dims, num_induc, num_induc)
    :returns:
        means of shape (num_samps, out_dims, time, 1)
        covariances of shape (num_samps, out_dims, time, time)
    """
    out_dims, num_induc, num_samps, ts = Kzx.shape

    eps_I = jitter * jnp.eye(num_induc)
    chol_Kzz = cholesky(Kzz + eps_I)  # (out_dims, num_induc, num_induc)
    Kzx = Kzx.reshape(out_dims, num_induc, -1)

    if whitened:
        v = u_mu
        L = u_Lcov
        W = solve_triangular(chol_Kzz, Kzx, lower=True)

    else:
        tpl = (u_mu, Kzx)
        if mean_only is False or compute_KL:
            tpl += (u_Lcov,)

        stacked = jnp.concatenate(
            tpl, axis=-1
        )  # (out_dims, num_induc, 1 + num_samps*ts + num_induc)
        Linv_stacked = solve_triangular(chol_Kzz, stacked, lower=True)

        v = Linv_stacked[..., :1]
        W = Linv_stacked[..., 1:-num_induc]
        if mean_only is False or compute_KL:
            L = Linv_stacked[..., -num_induc:]

    W = W.reshape(out_dims, num_induc, num_samps, ts).transpose(2, 0, 3, 1)
    post_means = W @ v[None, ...].repeat(
        num_samps, axis=0
    )  # (num_samps, out_dims, time, 1)

    if mean_only is False:
        WL = W @ L[None, ...].repeat(num_samps, axis=0)

        if diag_cov:
            post_covs = jnp.maximum(
                Kxx + (WL**2 - W**2).sum(-1, keepdims=True), 0.0
            )  # (num_samps, out_dims, time, 1)

        else:
            post_covs = (
                Kxx - W @ W.transpose(0, 1, 3, 2) + WL @ WL.transpose(0, 1, 3, 2)
            )  # (num_samps, out_dims, time, time)

    else:
        post_covs = None

    if compute_KL:
        trace_term = vtrace(L @ L.transpose(0, 2, 1)).sum()
        quadratic_form = (v.transpose(0, 2, 1) @ v).sum()
        log_determinants = -jnp.log(vdiag(L)).sum()

        KL = (
            0.5 * (trace_term + quadratic_form - out_dims * num_induc)
            + log_determinants
        )

    else:
        KL = 0.0

    aux = (chol_Kzz, W) if compute_aux else None
    return post_means, post_covs, KL, aux


def evaluate_tsparse_posterior(
    Kxx,
    Kzx,
    Kzz,
    lambda_1,
    chol_Lambda_2,
    mean_only,
    diag_cov,
    compute_KL,
    compute_aux,
    jitter,
):
    """
    :param jnp.ndarray Kxx: kernel (num_samps, out_dims, ts, ts) or None
    :param jnp.ndarray Kzx: kernel (num_samps, out_dims, num_induc, ts)
    :param jnp.ndarray Kzz: kernel (num_samps, out_dims, num_induc, num_induc)
    :returns:
        means of shape (out_dims, num_samps, time, 1)
        covariances of shape (out_dims, num_samps, time, time)
    """
    num_samps, out_dims, num_induc, ts = Kzx.shape

    eps_I = jitter * jnp.eye(num_induc)
    if mean_only is False or compute_KL or compute_aux:
        chol_Kzz = cholesky(Kzz + eps_I)  # (out_dims, num_induc, num_induc)

    induc_cov = chol_Lambda_2 @ chol_Lambda_2.transpose(0, 2, 1)
    chol_R = cholesky(Kzz + induc_cov)  # (out_dims, num_induc, num_induc)

    Kxz = Kzx.transpose(0, 1, 3, 2)  # (num_samps, out_dims, time, num_induc)

    Kxz_LRinv = solve_triangular(
        chol_R[None, ...].repeat(num_samps, axis=0), Kzx, lower=True
    ).transpose(
        0, 1, 3, 2
    )  # (num_samps, out_dims, time, num_induc)

    LRinv_lambda_1 = solve_triangular(chol_R, lambda_1, lower=True)

    post_means = Kxz_LRinv @ LRinv_lambda_1  # (num_samps, out_dims, time, 1)

    if mean_only is False or compute_aux:
        W = (
            solve_triangular(
                chol_Kzz,
                Kzx.transpose(1, 2, 0, 3).reshape(out_dims, num_induc, -1),
                lower=True,
            )
            .reshape(out_dims, num_induc, num_samps, ts)
            .transpose(2, 0, 3, 1)
        )

    if mean_only is False:
        if diag_cov:
            post_covs = jnp.maximum(
                Kxx
                - (W**2).sum(-1, keepdims=True)
                + (Kxz_LRinv**2).sum(-1, keepdims=True),
                0.0,
            )  # (num_samps, out_dims, time, 1)
        else:
            post_covs = (
                Kxx
                - W @ W.transpose(0, 1, 3, 2)
                + Kxz_LRinv @ Kxz_LRinv.transpose(0, 1, 3, 2)
            )  # (num_samps, out_dims, time, time)
    else:
        post_covs = None

    if compute_KL:
        LRinv_Lzz = solve_triangular(
            chol_R, chol_Kzz, lower=True
        )  # (out_dims, num_induc, num_induc)
        LRinv_Kzz_tLRinv = LRinv_Lzz @ LRinv_Lzz.transpose(0, 2, 1)

        trace_term = vtrace(LRinv_Kzz_tLRinv).sum()
        quadratic_form = (
            LRinv_lambda_1.transpose(0, 2, 1) @ LRinv_Kzz_tLRinv @ LRinv_lambda_1
        ).sum()
        log_determinants = -jnp.log(vdiag(LRinv_Lzz)).sum()

        KL = (
            0.5 * (trace_term + quadratic_form - out_dims * num_induc)
            + log_determinants
        )
    else:
        KL = 0.0

    aux = (chol_Kzz, W, Kxz_LRinv) if compute_aux else None
    return post_means, post_covs, KL, aux


def evaluate_lbfm_posterior(
    phi_sites,
    site_obs,
    site_Lcovs,
    mean_only,
    compute_KL,
):
    """
    Get posterior over weights in linear basis function model

    :param jnp.ndarray phi_sites: Gram matrix (out_dims, num_params, num_sites)
    :param jnp.ndarray site_obs: pseudo observations (out_dims, num_sites, 1)
    :param jnp.ndarray site_Lcovs: pseudo observation noise (out_dims, num_sites, num_sites)
    :returns:
        means of shape (out_dims, time, 1)
        covariances of shape (out_dims, time, time)
    """
    num_params = phi_sites.shape[1]
    obs_dims = site_obs.shape[0]
    Id = jnp.eye(num_params)

    Lphi = solve_triangular(
        site_Lcovs[:, None].repeat(num_params, axis=1), phi_sites[..., None]
    )[..., 0]

    M = Lphi @ Lphi.transpose(0, 2, 1) + Id
    L_prec = jnp.linalg.cholesky(M)

    PS = phi_sites @ site_obs
    v = solve_triangular(L_prec, PS, lower=True)
    post_means = solve_triangular(L_prec, v, lower=True)
    post_Lcovs = solve_triangular(
        L_prec, jnp.broadcast_to(Id, L_prec.shape), lower=True
    )

    if compute_KL:
        trace_term = vtrace(post_Lcovs @ post_Lcovs.transpose(0, 2, 1)).sum()
        quadratic_form = (post_means.transpose(0, 2, 1) @ post_means).sum()
        log_determinants = -jnp.log(vdiag(post_Lcovs)).sum()

        KL = (
            0.5 * (trace_term + quadratic_form - obs_dims * num_params)
            + log_determinants
        )

    return post_means, post_Lcovs, KL if compute_KL else 0.0

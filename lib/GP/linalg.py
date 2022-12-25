import math

import jax.numpy as jnp
from jax import lax, tree_map, vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import block_diag, cho_solve, solve_triangular

from ..utils.linalg import solve

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
        pseudo_mean, pseudo_Lcov, A, Q = inputs
        f_m, f_P, lZ = carry

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

        G = solve(predict_P, A_f_P).T  # G = F * A' * P^{-1} = (P^{-1} * A * F)'

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


def process_noise_covariance(A, Pinf):
    # Qₙ = Pinf - Aₙ Pinf Aₙ'
    Q = Pinf - A @ Pinf @ A.T
    return Q


def get_LTI_matrices(A, Pinf, timesteps):
    """
    Computes the transition and process noise matrices
    Handles both a time sequence and a single matrix input
    
    :param jnp.ndarray A: transition matrix of shape (sd, sd) or (ts, sd, sd)
    :param jnp.ndarray Pinf: stationary covariance of shape (sd, sd)
    """
    Id = jnp.eye(Pinf.shape[0])
    Zs = jnp.zeros_like(Pinf)
    if len(A.shape) == 2:  # single dt value, i.e. LTI
        Q = process_noise_covariance(A, Pinf)
        As = jnp.stack([Id] + [A] * (timesteps - 1) + [Id], axis=0)
        Qs = jnp.stack([Zs] + [Q] * (timesteps - 1) + [Zs], axis=0)
    else:
        Q = vmap(process_noise_covariance, (0, None), 0)(A, Pinf)
        As = jnp.concatenate((Id[None, ...], A, Id[None, ...]), axis=0)
        Qs = jnp.concatenate((Zs[None, ...], Q, Zs[None, ...]), axis=0)

    return As, Qs


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
    post_mean,
    post_cov,
    gain, 
    Pinf,
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
    Q_fwd = Pinf - A_fwd @ Pinf @ A_fwd.T
    Q_bwd = Pinf - A_bwd @ Pinf @ A_bwd.T
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
        return P @ mean_joint
    
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



def fixed_interval_smoothing(H, m0, P0, As, Qs, site_obs, site_Lcov, compute_logZ, return_gains):
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

    
    
def evaluate_LTI_posterior(
    H, minf, Pinf, As, Qs, site_obs, site_Lcov, interpolate_sites, mean_only, compute_KL, jitter
):
    """
    predict at test locations X, which may includes training points
    (which are essentially fixed inducing points)

    :param jnp.ndarray Pinf: stationary covariance (sd, sd)
    :param jnp.ndarray As: transition matrices of shape (locs, sd, sd)
    :return:
        means of shape (time, out, 1)
        covariances of shape (time, out, out)
    """    
    minf = minf[:, None]
    interpolate = (interpolate_sites is not None)

    smoother_means, smoother_covs, gains, logZ = fixed_interval_smoothing(
        H, minf, Pinf, As, Qs, site_obs, site_Lcov, compute_KL, interpolate)
    
    if compute_KL:  # compute using pseudo likelihood
        post_means = H @ smoother_means
        post_covs = H @ smoother_covs @ H.T
    elif interpolate_sites is None:  # evaluate at observed locs
        post_means = H @ smoother_means
        post_covs = None if mean_only else H @ smoother_covs @ H.T
        
    if compute_KL:
        site_log_lik = pseudo_log_likelihood(
            post_means, post_covs, site_obs, site_Lcov
        )
        KL = site_log_lik - logZ

    else:
        KL = 0.0

    if interpolate:  # predict the state distribution at the test time steps
        ind_eval, A_fwd, A_bwd = interpolate_sites
        
        predict_between_sites_vmap = vmap(
            predict_between_sites,
            (0, 0, 0, None, None, None, None, None, None),
            0 if mean_only else (0, 0),
        )  # vmap over eval_nums
        
        # add dummy states at either edge
        mean_aug = jnp.concatenate(
            [minf[None, :], smoother_means, minf[None, :]], axis=0
        )
        cov_aug = jnp.concatenate(
            [Pinf[None, ...], smoother_covs, Pinf[None, ...]], axis=0
        )
        gain_aug = jnp.concatenate([jnp.zeros_like(gains[:1, ...]), gains], axis=0)
        
        predicts = predict_between_sites_vmap(
            ind_eval, 
            A_fwd, 
            A_bwd, 
            mean_aug,
            cov_aug,
            gain_aug,
            Pinf,
            mean_only,
            jitter,
        )
        
        if mean_only:
            eval_mean = predicts
            post_means, post_covs = H @ eval_mean, None

        else:
            eval_mean, eval_cov = predicts
            post_means, post_covs = H @ eval_mean, H @ eval_cov @ H.T

    return post_means, post_covs, KL



# sparse
vdiag = vmap(jnp.diag)


def mvn_conditional(x, z, fz, kernel_func, mean_only, diag_cov, jitter):
    """
    Compute conditional of a MVN distribution p(f(x) | f(z))
    
    :param x jnp.ndarray: evaluation locations of shape (out_dims, eval_pts, x_dims)
    :param z jnp.ndarray: conditioning locations of shape (out_dims, cond_pts, x_dims)
    :param fz jnp.ndarray: conditioned observations of shape (out_dims, cond_pts, 1)
    """
    x_dims = x.shape[-1]
    
    Kxx = kernel_func(x, None, diag_cov)  # (out_dims, eval_pts, eval_pts)

    Kzz = kernel_func(z, None, False)  # (out_dims, cond_pts, cond_pts)
    eps_I = jitter * jnp.eye(x_dims)[None, ...]
    Lzz = cholesky(Kzz + eps_I)

    Kzx = kernel_func(z, x, False)  # (out_dims, cond_pts, eval_pts)

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
    kernel, induc_locs, mean, x, u_mu, u_Lcov, whitened, 
    mean_only, diag_cov, compute_KL, compute_aux, jitter
):
    """
    :param jnp.array x: input of shape (time, num_samps, in_dims)
    :returns:
        means of shape (out_dims, num_samps, time, 1)
        covariances of shape (out_dims, num_samps, time, time)
    """
    in_dims = kernel.in_dims
    out_dims = kernel.out_dims
    ts, num_samps = x.shape[:2]
    
    num_induc = induc_locs.shape[1]
    eps_I = jitter * jnp.eye(num_induc)[None, ...]

    Kzz = kernel.K(induc_locs, None, False)
    chol_Kzz = cholesky(Kzz + eps_I)  # (out_dims, num_induc, num_induc)
    
    Kzx = vmap(kernel.K, (None, 2, None), 1)(
        induc_locs, x[None, ...], False
    )  # (out_dims, num_samps, num_induc, time)

    if whitened:
        v = u_mu
        L = u_Lcov
        W = solve_triangular(
            chol_Kzz, Kzx.transpose(0, 2, 1, 3).reshape(out_dims, num_induc, -1), lower=True
        ).transpose(0, 2, 1).reshape(out_dims, num_samps, -1, num_induc)
        
    else:
        tpl = (u_mu, Kzx.transpose(0, 2, 1, 3).reshape(out_dims, num_induc, -1))
        if mean_only is False or compute_KL:
            tpl += (u_Lcov,)
            
        stacked = jnp.concatenate(tpl, axis=-1)  # (out_dims, num_induc, 1 + num_induc + num_samps*ts)
        Linv_stacked = solve_triangular(chol_Kzz, stacked, lower=True)
        
        v = Linv_stacked[..., :1]
        W = Linv_stacked[..., 1:-num_induc].transpose(0, 2, 1).reshape(
            out_dims, num_samps, -1, num_induc, 
        )
        if mean_only is False or compute_KL:
            L = Linv_stacked[..., -num_induc:]
    
    post_means = (
        W @ v[:, None, ...].repeat(num_samps, axis=1) + mean[:, None, None, None]
    )  # (out_dims, num_samps, time, 1)

    if mean_only is False:
        WL = W @ L[:, None, ...].repeat(num_samps, axis=1)
        
        Kxx = vmap(kernel.K, (2, None, None), 1)(
            x[None, ...], None, diag_cov, 
        )  # (out_dims, num_samps, time, time)
        
        if diag_cov:
            post_covs = (
                Kxx[..., jnp.arange(ts), jnp.arange(ts)][..., None] + \
                (WL**2 - W**2).sum(-1, keepdims=True)
            )  # (out_dims, num_samps, time, time)
            
        else:
            post_covs = (
                Kxx - W @ W.transpose(0, 1, 3, 2) + \
                WL @ WL.transpose(0, 1, 3, 2)
            )  # (out_dims, num_samps, time, 1)
            
    else:
        post_covs = None

    if compute_KL:
        if whitened:
            log_determinants = -jnp.log(vdiag(u_Lcov)).sum()
        else:
            log_determinants = (jnp.log(vdiag(chol_Kzz)) - jnp.log(vdiag(u_Lcov))).sum()

        trace_term = jnp.trace(L @ L.transpose(0, 2, 1)).sum()
        quadratic_form = (v.transpose(0, 2, 1) @ v).sum()
        KL = 0.5 * (trace_term + quadratic_form - num_induc) + log_determinants
        
    else:
        KL = 0.0

    aux = (chol_Kzz, W) if compute_aux else None
    return post_means, post_covs, KL, aux



def evaluate_tsparse_posterior(
    kernel, induc_locs, mean, x, lambda_1, chol_Lambda_2, 
    mean_only, diag_cov, compute_KL, compute_aux, jitter
):
    """
    :param jnp.array x: input of shape (time, num_samps, in_dims)
    :returns:
        means of shape (out_dims, num_samps, time, 1)
        covariances of shape (out_dims, num_samps, time, time)
    """
    in_dims = kernel.in_dims
    out_dims = kernel.out_dims
    ts, num_samps = x.shape[:2]

    num_induc = induc_locs.shape[1]
    eps_I = jitter * jnp.eye(num_induc)[None, ...]

    Kzz = kernel.K(induc_locs, None, False)
    if mean_only is False or compute_KL:
        chol_Kzz = cholesky(Kzz + eps_I)  # (out_dims, num_induc, num_induc)
        
    induc_cov = chol_Lambda_2 @ chol_Lambda_2.transpose(0, 2, 1)
    chol_R = cholesky(Kzz + induc_cov)  # (out_dims, num_induc, num_induc)

    Kzx = vmap(kernel.K, (None, 2, None), 1)(
        induc_locs, x[None, ...], False
    )  # (out_dims, num_samps, num_induc, time)
    Kxz = Kzx.transpose(0, 1, 3, 2)  # (out_dims, num_samps, time, num_induc)

    Kxz_Rinv = cho_solve(
        (chol_R[:, None, ...].repeat(num_samps, axis=1), True), Kzx
    ).transpose(0, 1, 3, 2)  # (out_dims, num_samps, time, num_induc)      
    
    post_means = (
        Kxz_Rinv @ lambda_1[:, None, ...].repeat(num_samps, axis=1)
        + mean[:, None, None, None]
    )  # (out_dims, num_samps, time, 1)

    if mean_only is False:
        Kxx = vmap(kernel.K, (2, None, None), 1)(
            x[None, ...], None, diag_cov
        )  # (out_dims, num_samps, time, time)
        
        W = solve_triangular(
            chol_Kzz, Kzx.transpose(0, 2, 1, 3).reshape(out_dims, num_induc, -1), lower=True
        ).transpose(0, 2, 1).reshape(out_dims, num_samps, -1, num_induc)

        if diag_cov:
            post_covs = (
                Kxx - (W**2).sum(-1, keepdims=True) + \
                (Kxz_Rinv * Kxz).sum(-1, keepdims=True)
            )  # (out_dims, num_samps, time, 1)
        else:
            post_covs = (
                Kxx - W @ W.transpose(0, 1, 3, 2) + Kxz_Rinv @ Kzx
            )  # (out_dims, num_samps, time, time)
    else:
        post_covs = None

    if compute_KL:
        stacked = jnp.concatenate((lambda_1, Kzz), axis=-1)
        Rinv_stacked = cho_solve((chol_R, True), stacked)
        Rinv_lambda_1 = Rinv_stacked[..., :1]  # (out_dims, num_induc, 1)
        Rinv_Kzz = Rinv_stacked[..., 1:]
        
        trace_term = jnp.trace(Rinv_Kzz).sum()
        quadratic_form = (
            lambda_1.transpose(0, 2, 1) @ Rinv_Kzz @ Rinv_lambda_1
        ).sum()
        log_determinants = (jnp.log(vdiag(chol_R)) - jnp.log(vdiag(chol_Kzz))).sum()
        KL = 0.5 * (trace_term + quadratic_form - num_induc) + log_determinants
    else:
        KL = 0.0

    if compute_aux:
        aux = (Kxz_Rinv,)  # squeeze shape
    else:
        aux = None

    return post_means, post_covs, KL, aux
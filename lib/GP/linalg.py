import jax.numpy as jnp
from jax import lax, tree_map, vmap
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import cho_solve, solve_triangular

from ..utils.linalg import solve





### LDS ###
def compute_kernel(delta_t, F, Pinf, H):
    """
    delta_t is positive and increasing
    """
    A = vmap(expm)(F[None, ...] * delta_t[:, None, None])
    At = vmap(expm)(-F.T[None, ...] * delta_t[:, None, None])
    P = (A[..., None] * Pinf[None, None, ...]).sum(-2)
    P_ = (Pinf[None, ..., None] * At[:, None, ...]).sum(-2)

    delta_t = np.broadcast_to(delta_t[:, None, None], P.shape)
    Kt = H[None, ...] @ jnp.where(delta_t > 0.0, P, P_) @ H.T[None, ...]
    return Kt


def discrete_transitions(F, L, Qc):
    """ """
    A = expm(F * dt)
    Pinf = solve_continuous_lyapunov(F, L @ L.T * Qc)
    Q = Pinf - A @ Pinf @ A.T
    return A, Pinf



# LGSSM
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
            n = solve_triangular(L, eta, lower=True)  # (state_dims, 1)
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



def compute_conditional_statistics(
    kernel_state_transition, Pinf, t_eval, t, ind, mean_only, jitter
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
    A_fwd = kernel_state_transition(dt_fwd)
    A_back = kernel_state_transition(dt_back)

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
    kernel_state_transition,
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
            kernel_state_transition, Pinf, t_eval, t, ind, mean_only, jitter
        )
        cross_cov = gain[ind] @ post_cov[ind + 1]
        cov_joint = jnp.block(
            [[post_cov[ind], cross_cov], [cross_cov.T, post_cov[ind + 1]]]
        )
        return P @ mean_joint, P @ cov_joint @ P.T + T

    else:
        P = compute_conditional_statistics(
            kernel_state_transition, Pinf, t_eval, t, ind, mean_only, jitter
        )
        return P @ mean_joint

    
    
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
    

    
def get_LGSSM_matrices(kernel, timedata, Pinf):
    t, dt = timedata
    Id = jnp.eye(Pinf.shape[0])
    Zs = jnp.zeros_like(Pinf)

    if dt.shape[0] == 1:
        A = kernel.state_transition(dt[0])
        Q = process_noise_covariance(A, Pinf)
        As = jnp.stack([Id] + [A] * (t.shape[0] - 1) + [Id], axis=0)
        Qs = jnp.stack([Zs] + [Q] * (t.shape[0] - 1) + [Zs], axis=0)
    else:
        As = vmap(kernel.state_transition)(dt)
        Qs = vmap(process_noise_covariance, (0, None), 0)(As, Pinf)
        As = jnp.concatenate((Id[None, ...], As, Id[None, ...]), axis=0)
        Qs = jnp.concatenate((Zs[None, ...], Qs, Zs[None, ...]), axis=0)

    return As, Qs
    
    
    
def evaluate_LGSSM_posterior(
    t_eval, As, Qs, H, minf, Pinf, kernel_state_transition, 
    t_obs, site_obs, site_Lcov, mean_only, compute_KL, jitter
):
    """
    predict at test locations X, which may includes training points
    (which are essentially fixed inducing points)

    :param jnp.ndarray t_eval: evaluation times of shape (locs,)
    :return:
        means of shape (time, out, 1)
        covariances of shape (time, out, out)
    """    
    minf = minf[:, None]

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
        t_aug = jnp.concatenate([-inf, t_obs, inf], axis=0)
        mean_aug = jnp.concatenate(
            [minf[None, :], smoother_means, minf[None, :]], axis=0
        )
        cov_aug = jnp.concatenate(
            [Pinf[None, ...], smoother_covs, Pinf[None, ...]], axis=0
        )
        gain_aug = jnp.concatenate([jnp.zeros_like(gains[:1, ...]), gains], axis=0)

        #in_shape = tree_map(lambda x: None, params["kernel"])
        predict_from_state_vmap = vmap(
            predict_from_state,
            (0, 0, None, None, None, None, None, None, None, None),
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
                kernel_state_transition, 
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
                kernel_state_transition,
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



# sparse
vdiag = vmap(jnp.diag)


def evaluate_sparse_posterior(
    kernel, induc_locs, mean, x, lambda_1, chol_Lambda_2, mean_only, compute_KL, compute_aux, jitter
):
    """
    :param jnp.array x: input of shape (time, num_samps, in_dims, 1)
    :returns:
        means of shape (out_dims, num_samps, time, 1)
        covariances of shape (out_dims, num_samps, time, time)
    """
    in_dims = kernel.in_dims
    out_dims = kernel.out_dims

    num_induc = induc_locs.shape[1]
    eps_I = jitter * jnp.eye(num_induc)[None, ...]

    Kzz = kernel.K(induc_locs, induc_locs)
    Kzz = jnp.broadcast_to(Kzz, (out_dims, num_induc, num_induc))
    induc_cov = chol_Lambda_2 @ chol_Lambda_2.transpose(0, 2, 1)
    chol_R = cholesky(Kzz + induc_cov)  # (out_dims, num_induc, num_induc)

    ts, num_samps = x.shape[:2]
    K = lambda x, y: kernel.K(x, y)

    Kxx = vmap(K, (2, 2), 1)(
        x[None, ...], x[None, ...]
    )  # (out_dims, num_samps, time, time)
    Kxx = jnp.broadcast_to(Kxx, (out_dims, num_samps, ts, ts))

    Kzx = vmap(K, (None, 2), 1)(
        induc_locs, x[None, ...]
    )  # (out_dims, num_samps, num_induc, time)
    Kzx = jnp.broadcast_to(Kzx, (out_dims, num_samps, num_induc, ts))
    Kxz = Kzx.transpose(0, 1, 3, 2)  # (out_dims, num_samps, time, num_induc)

    if mean_only is False or compute_KL:
        chol_Kzz = cholesky(Kzz + eps_I)  # (out_dims, num_induc, num_induc)
        Kxz_invKzz = cho_solve(
            (chol_Kzz[:, None, ...].repeat(num_samps, axis=1), True), Kzx
        ).transpose(0, 1, 3, 2)

    Rinv_lambda_1 = cho_solve((chol_R, True), self.lambda_1)  # (out_dims, num_induc, 1)
    post_means = (
        Kxz @ Rinv_lambda_1[:, None, ...].repeat(num_samps, axis=1)
        + mean[:, None, None, None]
    )  # (out_dims, num_samps, time, 1)

    if mean_only is False:
        invR_Kzx = cho_solve(
            (chol_R[:, None, ...].repeat(num_samps, axis=1), True), Kzx
        )  # (out_dims, num_samps, num_induc, time)
        post_covs = (
            Kxx - Kxz_invKzz @ Kzx + Kxz @ invR_Kzx
        )  # (out_dims, num_samps, time, time)
    else:
        post_covs = None

    if compute_KL:
        trace_term = jnp.trace(cho_solve((chol_R, True), Kzz)).sum()
        quadratic_form = (
            Rinv_lambda_1.transpose(0, 2, 1) @ Kzz @ Rinv_lambda_1
        ).sum()
        log_determinants = (jnp.log(vdiag(chol_R)) - jnp.log(vdiag(chol_Kzz))).sum()
        KL = 0.5 * (trace_term + quadratic_form - num_induc) + log_determinants
    else:
        KL = 0.0

    if compute_aux:
        aux = (Kxz, Kxz_invKzz, chol_R)  # squeeze shape
    else:
        aux = None

    return post_means, post_covs, KL, aux
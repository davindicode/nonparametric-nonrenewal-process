import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import circmean, circvar
from torch.nn.parameter import Parameter

sys.path.append("../..")


from tqdm.autonotebook import tqdm


# selecting runs
def class_x_t(x_t, margin=10):
    base_low_thresh = 75
    base_high_thresh = 275
    low_thresh = base_low_thresh
    high_thresh = base_high_thresh
    prev_class = 0
    class_x_t = np.zeros_like(x_t)

    for i, x in enumerate(x_t):
        if x > high_thresh:
            new_class = 1
            high_thresh = base_high_thresh - margin
        elif x < low_thresh:
            new_class = -1
            low_thresh = base_low_thresh + margin
        else:
            new_class = 0
            if prev_class == -1:
                low_thresh = base_low_thresh - margin
                high_thresh = base_high_thresh
            elif prev_class == 1:
                high_thresh = base_high_thresh + margin
                low_thresh = base_low_thresh
            else:
                pass

        class_x_t[i] = new_class
        prev_class = new_class

    return class_x_t


def L_R_run(class_x_t):
    # Use directions -1, 0, 1 for R-L, none and L-R respectively
    prev_dir = 0
    prev_state = 0
    dir_t = np.zeros_like(class_x_t)

    incomp_run_count = 0

    for i, state in enumerate(class_x_t):
        if state == prev_state:
            dir_t[i] = prev_dir
        elif prev_state == -1 and state == 0:
            # Starting L-R run
            dir_t[i] = 1
        elif prev_state == 1 and state == 0:
            # Starting R-L run
            dir_t[i] = -1
        elif prev_state == 0 and prev_dir == 1:
            if state == -1:
                # Didn't complete run. This is a problem
                incomp_run_count += 1
                dir_t[i] = 0
            else:  # state == 1
                # Completed L-R run
                dir_t[i] = 0
        elif prev_state == 0 and prev_dir == -1:
            if state == 1:
                # Didn't complete run. This is a problem
                incomp_run_count += 1
                dir_t[i] = 0
            else:  # state == -1
                # Completed R-L run
                dir_t[i] = 0

        prev_dir = dir_t[i]
        prev_state = state

    print("# incompleted runs: ", incomp_run_count)
    return dir_t


def new_L_R_run(x_t, margin=10):
    "L-R: 0, R-L:1, Stationary bottom: 2, Stationary top: 3"
    base_low_thresh = 75
    base_high_thresh = 275
    low_thresh = base_low_thresh
    high_thresh = base_high_thresh
    prev_class = 2
    class_x_t = np.zeros_like(x_t)

    for i, x in enumerate(x_t):
        if x > high_thresh:
            new_class = 3
            high_thresh = base_high_thresh - margin
        elif x < low_thresh:
            new_class = 2
            low_thresh = base_low_thresh + margin
        else:
            if prev_class == 0 or prev_class == 1:
                new_class = prev_class
            elif prev_class == 2:
                new_class = 0
                low_thresh = base_low_thresh - margin
                high_thresh = base_high_thresh
            elif prev_class == 3:
                new_class = 1
                high_thresh = base_high_thresh + margin
                low_thresh = base_low_thresh
            else:
                pass

        class_x_t[i] = new_class
        prev_class = new_class

    return class_x_t


# Theta phase properties
def LWL_model(p, sample_bin, behav_tuple, spiketrain, kernel_func):
    r"""
    Perform smoothing using locally weighted maximum likelihood estimation [1], which is applied to 
    hippocampal data (2D position and theta phase input as covariates) to obtain phase fields.
    
    .. math::
            [f, u] &\sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
            y & \sim p(y) = p(y \mid f) p(f),
    
    References:
    
    [1] `Spike train dynamics predicts theta-related phase precession in hippocampal pyramidal cells`,
    Kenneth D. Harris, Darrell A. Henze, Hajime Hirase, Xavier Leinekugel, George Dragoi, 
    Andras Czurko & Gyorgy Buzsaki
    
    :param torch.tensor p: positions to evaluate the fields, has shape (dim, pos_dims)
    :param string name: name the colormap
    :returns: evaluated field (rate, theta, beta) tensors of shape (units, pos)
    :rtype: tuple
    """
    prevshape = p.shape[1:]
    p = p.reshape(p.shape[0], -1).T

    x_t, y_t, theta_t = behav_tuple
    p_t = np.concatenate((x_t[:, None], y_t[:, None]), axis=1)[
        None, ...
    ]  # pos, timestep, dim

    phase_field = []
    beta_field = []
    for trn in spiketrain:
        spiketimes = trn.nonzero()[0]
        theta_s = theta_t[spiketimes]

        kern = kernel_func(p[:, None, :], p_t[:, spiketimes, :])  # pos, timestep
        x = np.exp(theta_s * 1j)[None, :]
        x_sum = (x * (kern * trn[None, spiketimes])).sum(-1) / (
            kern * trn[None, spiketimes]
        ).sum(-1)
        phase_field.append(WrapPi(np.angle(x_sum), True))
        R = np.abs(x_sum)
        beta_field.append(R * (2 - R**2) / (1 - R**2))

        # print(kern.shape)

    phase_field = np.array(phase_field)
    beta_field = np.array(beta_field)
    return phase_field.reshape(-1, *prevshape), beta_field.reshape(-1, *prevshape)


def phase_field(sample_bin, covariates, spat_bins, spiketimes):
    r"""
    Compute the phase field by computing the moments of the local phases, a locally weighted circular mean [1].

    References:
    [1] ``, Fiser

    :param float sample_bin:
    :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
    :rtype: tuple(torch.Tensor, torch.Tensor)
    """
    dim = len(spat_bins)
    bin_count = [len(bins) - 1 for bins in spat_bins]
    A = []
    for times in spiketimes:
        c = [cov[times] for cov in covariates]
        samples = len(times)

        if dim == 1:
            a = [[] for l in range(bin_count[0])]
        elif dim == 2:
            a = [[[] for l in range(bin_count[1])] for k in range(bin_count[0])]
        elif dim == 3:
            a = [
                [[[] for l in range(bin_count[2])] for k in range(bin_count[1])]
                for r in range(bin_count[0])
            ]
        else:
            raise NotImplementedError

        tg = []
        for k, cov in enumerate(c[:-1]):
            tg.append(np.digitize(cov, spat_bins[k]) - 1)

        for t in range(samples):
            b = a
            for d in range(dim):
                b = b[tg[d][t]]
            b.append(c[-1][t])

        A.append(a)

    A = np.array(A)
    A_ = A.reshape(-1)
    mu = []
    var = []

    for n, a_ in enumerate(A_):
        mu.append(circmean(np.array(a_), low=0, high=2 * np.pi))
        var.append(circvar(np.array(a_), low=0, high=2 * np.pi))

    mu = np.array(mu).reshape(*A.shape)
    var = np.array(var).reshape(*A.shape)
    return A, mu, var


def fit_oscillatory_ISI(
    sample_bin,
    ISI,
    bins,
    lr=1e-3,
    max_iters=1000,
    loss_tolerance=0,
    tolerance_steps=10,
    dev="cpu",
):
    """
    Fit the ISI distribution with a Gamma function that has oscillatory modulation
    ISI is in ms, while sample_bin is in s

    :param float sample_bin: length of a time bin
    :param list ISI: list of numpy arrays giving ISI values, each list element is a neuron
    :param int bins:
    :param float lr: learning rate for optimization
    :param string dev: device to compute on
    :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
    :rtype: tuple(torch.Tensor, torch.Tensor)
    """
    units = len(ISI)
    x = np.arange(0, bins) * sample_bin
    p = []
    for u in range(units):
        points, _ = np.histogram(ISI[u], x, density=True)
        p.append(points)
    p = np.array(p)

    # Fit the theta period
    omega = Parameter(
        torch.empty((units, 1), device=dev).fill_(1.0 / (500.0 * sample_bin))
    )
    phi = Parameter(torch.empty((units, 1), device=dev).fill_(0.0))
    amp = Parameter(torch.empty((units, 1), device=dev).fill_(1.0))
    tau = Parameter(torch.empty((units, 1), device=dev).fill_(bins * sample_bin))
    alpha = Parameter(torch.empty((units, 1), device=dev).fill_(1.0))
    modamp = Parameter(torch.empty((units, 1), device=dev).fill_(1.0))

    optimizer = optim.Adam([omega, phi, amp, tau, alpha, modamp], lr=lr)

    x_tens = torch.tensor(x[:-1], device=dev).unsqueeze(0)
    y_tens = torch.tensor(p[:, :], device=dev)

    cnt = 0
    p_loss = 0
    iterator = tqdm(range(max_iters))
    for k in iterator:
        optimizer.zero_grad()
        acg_func = (
            amp
            * ((x_tens + 1e-12) / tau) ** (alpha - 1)
            * torch.exp(-x_tens / tau)
            * (1 + modamp * torch.cos(omega * x_tens + phi))
        )

        loss = (acg_func - y_tens).pow(2).mean()
        loss.backward()
        optimizer.step()
        modamp.data = torch.clamp(modamp.data, min=0, max=1)
        alpha.data = torch.clamp(alpha.data, min=0)
        tau.data = torch.clamp(tau.data, min=1e-12)
        if p_loss - loss.item() < loss_tolerance:
            cnt += 1
            if cnt > tolerance_steps:
                break
        else:
            cnt = 0
        p_loss = loss.item()
        iterator.set_postfix(loss=p_loss)

    omega_p = omega.data.squeeze(1).cpu().numpy()
    phi_p = phi.data.squeeze(1).cpu().numpy()
    amp_p = amp.data.squeeze(1).cpu().numpy()
    tau_p = tau.data.squeeze(1).cpu().numpy()
    alpha_p = alpha.data.squeeze(1).cpu().numpy()
    modamp_p = modamp.data.squeeze(1).cpu().numpy()

    param_tuple = (omega_p, phi_p, amp_p, tau_p, alpha_p, modamp_p)

    return x[:-1], acg_func.data.cpu().numpy(), p, param_tuple


def EEG_fit(
    sample_bin,
    eeg_t,
    lag_range=1250,
    time_window=1000000,
    lr=1e-3,
    max_iters=1000,
    loss_tolerance=0,
    tolerance_steps=10,
    dev="cpu",
):
    r"""
    EEG period fitting, fit the autocorrelation period by exponentially decaying oscillations
    """

    lag = np.arange(lag_range)
    corr = np.empty(lag_range)
    refs = eeg_t[0:time_window] - eeg_t[0:time_window].mean()
    refs_std = refs.std()
    for k in lag:
        coms = eeg_t[k : k + time_window] - eeg_t[k : k + time_window].mean()
        coms_std = coms.std()
        corr[k] = np.inner(refs, coms) / time_window / refs_std / coms_std

    omega = Parameter(torch.tensor(0.1, device=dev))
    phi = Parameter(torch.tensor(0.0, device=dev))
    amp = Parameter(torch.tensor(1.0, device=dev))
    tau = Parameter(torch.tensor(100.0, device=dev))
    power = Parameter(torch.tensor(-1.0, device=dev))
    offs = Parameter(torch.tensor(0.0, device=dev))
    toffs = Parameter(torch.tensor(1.0, device=dev))
    optimizer = optim.Adam(
        [omega] + [phi] + [amp] + [tau] + [power] + [toffs] + [offs], lr=lr
    )
    lag_tens = torch.tensor(lag, device=dev)
    corr_tens = torch.tensor(corr, device=dev)

    cnt = 0
    p_loss = 0
    iterator = tqdm(range(max_iters))
    for k in iterator:
        optimizer.zero_grad()
        acg_func = (
            amp
            * (lag_tens / tau + toffs) ** (power)
            * torch.cos(omega * lag_tens + phi)
            + offs
        )
        loss = (acg_func - corr_tens).pow(2).sum()
        loss.backward()
        optimizer.step()
        tau.data = torch.clamp(tau.data, min=1e-12)
        toffs.data = torch.clamp(toffs.data, min=1e-12)

        if p_loss - loss.item() < loss_tolerance:
            cnt += 1
            if cnt > tolerance_steps:
                break
        else:
            cnt = 0
        p_loss = loss.item()
        iterator.set_postfix(loss=p_loss)

    fit_p = acg_func.data.cpu().numpy()
    omega_p = omega.data.cpu().numpy() / sample_bin
    tau_p = tau.data.cpu().numpy() * sample_bin
    power_p = power.data.cpu().numpy()

    param_tuple = (omega_p, tau_p, power_p)

    return corr, fit_p, param_tuple

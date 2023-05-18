import numpy as np

import scipy.special as sps
import scipy.stats as scstats
from scipy import signal


# histograms
def occupancy_normalized_histogram(
    sample_bin, time_thres, covariates, cov_bins, activities=None, spiketimes=None
):
    """
    Compute the occupancy-normalized activity histogram for neural data, corresponding to maximum
    likelihood estimation of the rate in an inhomogeneous Poisson process.

    :param float sample_bin: binning time
    :param float time_thres: only count bins with total occupancy time above time_thres
    :param list covariates: list of time series that describe animal behaviour
    :param tuple cov_bins: tuple of arrays describing bin boundary locations
    :param list spiketimes: list of arrays containing spike times of neurons
    :param bool divide: return the histogram divisions (rate and histogram probability in region)
                        or the activity and time histograms
    """
    if spiketimes is not None:
        units = len(spiketimes)
        sg_c = [() for u in range(units)]
    else:
        units = activities.shape[0]

    c_bins = ()
    tg_c = ()
    for k, cov in enumerate(covariates):
        c_bins += (len(cov_bins[k]) - 1,)
        tg_c += (np.digitize(cov, cov_bins[k]) - 1,)
        if spiketimes is not None:
            for u in range(units):
                sg_c[u] += (np.digitize(cov[spiketimes[u]], cov_bins[k]) - 1,)

    # get time spent in each bin
    occupancy_time = np.zeros(tuple(len(bins) - 1 for bins in cov_bins))
    np.add.at(occupancy_time, tg_c, sample_bin)

    # get activity of each bin per neuron
    tot_activity = np.zeros((units,) + tuple(len(bins) - 1 for bins in cov_bins))
    if spiketimes is not None:
        for u in range(units):
            np.add.at(tot_activity[u], sg_c[u], 1)
    else:
        for u in range(units):
            np.add.at(tot_activity[u], tg_c, activities[u, :])

    avg_activity = np.zeros((units,) + tuple(len(bins) - 1 for bins in cov_bins))
    occupancy_time[occupancy_time <= time_thres] = -1.0  # avoid division by zero
    for u in range(units):
        tot_activity[u][occupancy_time < time_thres] = 0.0
        avg_activity[u] = tot_activity[u] / occupancy_time
    occupancy_time[
        occupancy_time < time_thres
    ] = 0.0  # take care of unvisited/uncounted bins

    return avg_activity, occupancy_time, tot_activity


def smooth_histogram(hist_values, sm_filter, padding_modes):
    r"""
    Neurons is the batch dimension, parallelize the convolution for 1D, 2D or 3D
    The padding modes are ('periodic', 'repeat', 'zeros', 'none')
    sm_filter should had odd sizes for its shape

    :param np.ndarray hist_values: input histogram array of shape (out_dims, ndim_1, ndim_2, ...)
    :param np.ndarray sm_filter: input filter array of shape (ndim_1, ndim_2, ...)
    :param list padding_modes: list of strings (per dimension) to indicate convolution boundary conditions
    :returns:
        smoothened histograms np.ndarray
    """
    for s in sm_filter.shape:
        assert s % 2 == 1  # odd shape sizes

    out_dims = hist_values.shape[0]
    dim = len(hist_values.shape) - 1
    assert dim > 0
    step_sm = np.array(sm_filter.shape) // 2

    for d in range(dim):
        if d > 0:
            hist_values = np.swapaxes(hist_values, 1, d + 1)

        if padding_modes[d] == "repeat":
            hist_values = np.concatenate(
                (
                    np.repeat(hist_values[:, :1, ...], step_sm[d], axis=1),
                    hist_values,
                    np.repeat(hist_values[:, -1:, ...], step_sm[d], axis=1),
                ),
                axis=1,
            )

        elif padding_modes[d] == "periodic":
            hist_values = np.concatenate(
                (
                    hist_values[:, -step_sm[d] :, ...],
                    hist_values,
                    hist_values[:, : step_sm[d], ...],
                ),
                axis=1,
            )

        elif bopadding_modesund[d] == "zeros":
            zz = np.zeros_like(hist_values[:, : step_sm[d], ...])
            hist_values = np.concatenate((zz, hist_values, zz), axis=1)

        elif padding_modes[d] != "none":
            raise ValueError("Invalid padding mode for array bounds")

        if d > 0:
            hist_values = np.swapaxes(hist_values, 1, d + 1)

    hist_smth = []
    for u in range(out_dims):
        hist_smth.append(signal.convolve(hist_values[u], sm_filter, mode="valid"))
    hist_smth = np.array(hist_smth)

    return hist_smth


def traverse_histogram(x, input_bins, histogram_weights):
    """
    :param np.ndarray x: inputs of shape (ts, in_dims)
    :param no.ndarray histogram_weights: histogram weights (out_dims, )
    """
    out_dims = histogram_weights.shape[0]
    ts, in_dims = x.shape

    indices = []
    for k in range(in_dims):
        bins = input_bins[k]
        a = np.searchsorted(bins, x[:, k], side="right")
        indices.append(a - 1)

    hist_ind = (np.arange(out_dims)[:, None].repeat(ts, axis=1),) + tuple(
        ind[None, :].repeat(out_dims, axis=0) for ind in indices
    )

    return histogram_weights[hist_ind]


def KDE_trajectory(bins_tuple, covariates, sm_size, L, smooth_modes):
    """
    Kernel density estimation of covariate time series, with Gaussian kernels.
    """
    dim = len(bins_tuple)
    assert (
        (dim == len(covariates))
        and (dim == len(sm_size))
        and (dim == len(smooth_modes))
    )
    time_samples = covariates[0].shape[0]
    c_bins = ()
    tg_c = ()
    for k, cov in enumerate(covariates):
        c_bins += (len(bins_tuple[k]) - 1,)
        tg_c += (np.digitize(cov, bins_tuple[k]) - 1,)

    # get time spent in each bin
    bin_time = np.zeros(tuple(len(bins) - 1 for bins in bins_tuple))
    np.add.at(bin_time, tg_c, 1)
    bin_time /= bin_time.sum()  # normalize

    sm_centre = np.array(sm_size) // 2
    sm_filter = np.ones(tuple(sm_size))
    ones_arr = np.ones_like(sm_filter)
    for d in range(dim):
        size = sm_size[d]
        centre = sm_centre[d]
        L_ = L[d]

        if d > 0:
            bin_time = np.swapaxes(bin_time, 0, d)
            ones_arr = np.swapaxes(ones_arr, 0, d)
            sm_filter = np.swapaxes(sm_filter, 0, d)

        for k in range(size):
            sm_filter[k, ...] *= (
                np.exp(-0.5 * (((k - centre) / L_) ** 2)) * ones_arr[k, ...]
            )

        if d > 0:
            bin_time = np.swapaxes(bin_time, 0, d)
            ones_arr = np.swapaxes(ones_arr, 0, d)
            sm_filter = np.swapaxes(sm_filter, 0, d)

    smth_time = smooth_hist(bin_time[None, ...], sm_filter, smooth_modes)
    smth_time /= smth_time.sum()  # normalize
    return smth_time[0, ...], bin_time


# percentiles
def percentiles_from_samples(samples, percentiles=[0.05, 0.5, 0.95]):
    """
    Compute quantile intervals from samples

    :param np.ndarray samples: input samples of shape (MC, ...)
    :param list percentiles: list of percentile values to look at
    :returns:
        list of tensors representing percentile boundaries
    """
    num_samples = samples.shape[0]
    samples = np.sort(samples, axis=0)  # sort for percentiles

    percentile_samples = [
        samples[int(num_samples * percentile)] for percentile in percentiles
    ]
    return percentile_samples


def percentiles_transformed_gaussian(
    means, variances, inv_link, MC, percentiles=[0.05, 0.5, 0.95], rng=None
):
    """
    Sample from univariate Gaussians mapped through some link function

    :returns:
        transformed samples from the univariate Gaussians
    """
    if rng is None:
        rng = np.random.default_rng(123)

    samples = means + np.sqrt(variances) * rng.normal(size=(MC,) + means.shape)
    samples = inv_link(samples)  # (mc, ...)
    return percentiles_from_samples(samples, percentiles)


# KS and dispersion statistics
def counts_to_quantiles(P_count, counts, rng):
    """
    :param np.array P_count: count distribution values (ts, counts)
    :param np.array counts: count values (ts,)
    """
    counts = counts.astype(int)
    deq_noise = rng.uniform(size=counts.shape)

    cumP = np.cumsum(P_count, axis=-1)  # T, K
    tt = np.arange(counts.shape[0])
    quantiles = cumP[tt, counts] - P_count[tt, counts] * deq_noise
    return quantiles


def quantile_Z_mapping(x, inverse=False, LIM=1e-15):
    """
    Transform to Z-scores from quantiles in forward mapping.
    """
    if inverse:
        Z = x
        q = scstats.norm.cdf(Z)
        return q
    else:
        q = x
        _q = 1.0 - q
        _q[_q < LIM] = LIM
        _q[_q > 1.0 - LIM] = 1.0 - LIM
        Z = scstats.norm.isf(_q)
        return Z


def KS_sampling_dist(x, samples, K=100000):
    """
    Sampling distribution for the Brownian bridge supremum (KS test sampling distribution).
    """
    k = np.arange(1, K + 1)[None, :]
    return (
        8
        * x
        * (
            (-1) ** (k - 1) * k**2 * np.exp(-2 * k**2 * x[:, None] ** 2 * samples)
        ).sum(-1)
        * samples
    )


def KS_test(quantiles, alpha=0.05):
    """
    Kolmogorov-Smirnov and dispersion statistics using quantiles.
    """
    samples = quantiles.shape[0]
    assert samples > 1

    q_order = np.append(np.array([0]), np.sort(quantiles))
    ks_y = np.arange(samples + 1) / samples - q_order

    T_KS = np.abs(ks_y).max()
    z = quantile_Z_mapping(quantiles)
    sign_KS = np.sqrt(-0.5 * np.log(alpha)) / np.sqrt(samples)
    p_KS = np.exp(-2 * samples * T_KS**2)
    return q_order, T_KS, sign_KS, p_KS


def DS_test(quantiles, alpha=0.05):
    """
    Dispersion statistic and null distribution test
    """
    samples = quantiles.shape[0]
    assert samples > 1

    z = quantile_Z_mapping(quantiles)
    T_DS = np.log((z**2).mean()) + 1 / samples + 1 / 3 / samples**2
    T_DS_ = T_DS / np.sqrt(2 / (samples - 1))  # unit normal null distribution

    sign_DS = sps.erfinv(1 - alpha / 2.0) * np.sqrt(2) * np.sqrt(2 / (samples - 1))
    # ref_sign_DS = sps.erfinv(1-alpha_s/2.)*np.sqrt(2)
    # T_DS /= ref_sign_DS
    # sign_DS /= ref_sign_DS
    p_DS = 2.0 * (1 - scstats.norm.cdf(T_DS_))

    return T_DS, sign_DS, p_DS

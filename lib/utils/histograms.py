class histogram(base._input_mapping):
    """
    Histogram rate model based on GLM framework.
    Has an identity link function with positivity constraint on the weights.
    Only supports regressor mode.
    """

    def __init__(
        self,
        bins_cov,
        out_dims,
        ini_rate=1.0,
        alpha=None,
        tensor_type=torch.float,
        active_dims=None,
    ):
        """
        The initial rate should not be zero, as that gives no gradient in the Poisson
        likelihood case

        :param tuple bins_cov: tuple of bin objects (np.linspace)
        :param int neurons: number of neurons in total
        :param float ini_rate: initial rate array
        :param float alpha: smoothness prior hyperparameter, None means no prior
        """
        super().__init__(len(bins_cov), out_dims, tensor_type, active_dims)
        ini = torch.tensor([ini_rate]).view(-1, *np.ones(len(bins_cov)).astype(int))
        self.register_parameter(
            "w",
            Parameter(
                ini
                * torch.ones(
                    (out_dims,) + tuple(len(bins) - 1 for bins in bins_cov),
                    dtype=self.tensor_type,
                )
            ),
        )
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=self.tensor_type))
        else:
            self.alpha = alpha
        self.bins_cov = bins_cov  # use PyTorch for integer indexing

    def set_params(self, w=None):
        if w is not None:
            self.w.data = torch.tensor(w, device=self.w.device, dtype=self.tensor_type)

    def compute_F(self, XZ):
        XZ = self._XZ(XZ)
        samples = XZ.shape[0]

        tg = []
        for k in range(self.input_dims):
            tg.append(torch.bucketize(XZ[..., k], self.bins_cov[k], right=True) - 1)

        XZ_ind = (
            torch.arange(samples)[:, None, None].expand(
                -1, self.out_dims, len(tg[0][b])
            ),
            torch.arange(self.out_dims)[None, :, None].expand(
                samples, -1, len(tg[0][b])
            ),
        ) + tuple(tg_[b][:, None, :].expand(samples, self.out_dims, -1) for tg_ in tg)

        return self.w[XZ_ind], 0

    def sample_F(self, XZ):
        return self.compute_F(XZ)[0]

    def KL_prior(self, importance_weighted):
        if self.alpha is None:
            return super().KL_prior(importance_weighted)

        smooth_prior = (
            self.alpha[0] * (self.w[:, 1:, ...] - self.w[:, :-1, ...]).pow(2).sum()
            + self.alpha[1] * (self.w[:, :, 1:, :] - self.w[:, :, :-1, :]).pow(2).sum()
            + self.alpha[2] * (self.w[:, ..., 1:] - self.w[:, ..., :-1]).pow(2).sum()
        )
        return -smooth_prior

    def constrain(self):
        self.w.data = torch.clamp(self.w.data, min=0)

    def set_unvisited_bins(self, ini_rate=1.0, unvis=np.nan):
        """
        Set the bins that have not been updated to unvisited value unvis.
        """
        self.w.data[self.w.data == ini_rate] = torch.tensor(unvis, device=self.w.device)


def spike_threshold(
    sample_bin, bin_thres, covariates, cov_bins, spiketimes, direct=False
):
    """
    Only include spikes that correspond to bins with sufficient occupancy time. This is useful
    when using histogram models to avoid counting undersampled bins, which suffer from huge
    variance when computing the average firing rates in a histogram.

    :param float sample_bin: binning time
    :param float bin_thres: only count bins with total occupancy time above bin_thres
    :param list covariates: list of time series that describe animal behaviour
    :param tuple cov_bins: tuple of arrays describing bin boundary locations
    :param list spiketimes: list of arrays containing spike times of neurons
    """
    units = len(spiketimes)
    c_bins = ()
    tg_c = ()
    sg_c = [() for u in range(units)]
    for k, cov in enumerate(covariates):
        c_bins += (len(cov_bins[k]) - 1,)
        tg_c += (np.digitize(cov, cov_bins[k]) - 1,)
        for u in range(units):
            sg_c[u] += (np.digitize(cov[spiketimes[u]], cov_bins[k]) - 1,)

    # get time spent in each bin
    bin_time = np.zeros(tuple(len(bins) - 1 for bins in cov_bins))
    np.add.at(bin_time, tg_c, sample_bin)

    # get activity of each bin per neuron
    activity = np.zeros((units,) + tuple(len(bins) - 1 for bins in cov_bins))
    a = np.where(bin_time <= bin_thres)  # delete spikes in thresholded bins

    # get flattened removal indices
    remove_glob_index = a[0]
    fac = 1
    for k, cc in enumerate(c_bins[:-1]):
        fac *= cc
        remove_glob_index += a[k + 1] * fac

    # get flattened spike indices
    thres_spiketimes = []
    for u in range(units):
        s_glob_index = sg_c[u][0]
        fac = 1
        for k, cc in enumerate(c_bins[:-1]):
            fac *= cc
            s_glob_index += sg_c[u][k + 1] * fac

        rem_ind = np.array([], dtype=np.int64)
        for rg in remove_glob_index:
            rem_ind = np.concatenate((rem_ind, np.where(s_glob_index == rg)[0]))

        t_spike = np.delete(spiketimes[u], rem_ind)
        thres_spiketimes.append(t_spike)

    return thres_spiketimes


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


# mutual information
def spike_var_MI(rate, prob):
    """
    Mutual information analysis for inhomogeneous Poisson process rate variable.
    Uses histgrams to estimate the probability densities.

    .. math::
            I(x;\text{spike}) = \int p(x) \, \lambda(x) \, \log{\frac{\lambda(x)}{\langle \lambda \rangle}} \, \mathrm{d}x,

    :param np.array rate: rate variables of shape (neurons, covariate_dims...)
    :param np.array prob: occupancy values for each bin of shape (covariate_dims...)
    """
    units = rate.shape[0]

    MI = np.empty(units)
    logterm = rate / (
        (rate * prob[np.newaxis, :]).sum(
            axis=tuple(k for k in range(1, len(rate.shape))), keepdims=True
        )
        + 1e-12
    )
    logterm[logterm == 0] = 1.0  # goes to zero in log terms
    for u in range(units):  # MI in bits
        MI[u] = (prob * rate[u] * np.log2(logterm[u])).sum()

    return MI


def var_var_MI(sample_bin, v1_t, v2_t, v1_bin, v2_bin):
    """
    MI analysis between covariates
    Uses histograms for density estimation
    """

    # Dirichlet prior on the empirical variable probabilites TODO? Bayesian point estimate

    # binning
    track_samples = len(v1_t)
    v1_bins = len(v1_bin) - 1
    v2_bins = len(v2_bin) - 1
    bv_1 = np.digitize(v1_t, v1_bin) - 1
    bv_2 = np.digitize(v2_t, v2_bin) - 1

    # get empirical probability distributions
    p_12 = np.zeros((v1_bins, v2_bins))
    np.add.at(p_12, (bv_1, bv_2), 1.0 / track_samples)

    logterm = p_12 / (p_12.sum(0, keepdims=True) * p_12.sum(1, keepdims=True) + 1e-12)
    logterm[logterm == 0] = 1.0
    return (p_12 * np.log2(logterm)).sum()


# histograms
def smooth_hist(rate_binned, sm_filter, bound, dev="cpu"):
    r"""
    Neurons is the batch dimension, parallelize the convolution for 1D, 2D or 3D
    bound indicates the padding mode ('periodic', 'repeat', 'zeros')
    sm_filter should had odd sizes for its shape

    :param np.array rate_binned: input histogram array of shape (units, ndim_1, ndim_2, ...)
    :param np.array sm_filter: input filter array of shape (ndim_1, ndim_2, ...)
    :param list bound: list of strings (per dimension) to indicate convolution boundary conditions
    :param string dev: device to perform convolutions on
    :returns: smoothened histograms
    :rtype: np.array
    """
    for s in sm_filter.shape:
        assert s % 2 == 1  # odd shape sizes
    units = rate_binned.shape[0]
    dim = len(rate_binned.shape) - 1
    assert dim > 0
    step_sm = np.array(sm_filter.shape) // 2

    for d in range(dim):
        if d > 0:
            rate_binned = np.swapaxes(rate_binned, 1, d + 1)
        if bound[d] == "repeat":
            rate_binned = np.concatenate(
                (
                    np.repeat(rate_binned[:, :1, ...], step_sm[d], axis=1),
                    rate_binned,
                    np.repeat(rate_binned[:, -1:, ...], step_sm[d], axis=1),
                ),
                axis=1,
            )
        elif bound[d] == "periodic":
            rate_binned = np.concatenate(
                (
                    rate_binned[:, -step_sm[d] :, ...],
                    rate_binned,
                    rate_binned[:, : step_sm[d], ...],
                ),
                axis=1,
            )
        elif bound[d] == "zeros":
            zz = np.zeros_like(rate_binned[:, : step_sm[d], ...])
            rate_binned = np.concatenate((zz, rate_binned, zz), axis=1)
        else:
            raise NotImplementedError

        if d > 0:
            rate_binned = np.swapaxes(rate_binned, 1, d + 1)

    smth_rate = []
    for u in range(units):
        smth_rate.append(signal.convolve(rate_binned[u], sm_filter, mode="valid"))
    smth_rate = np.array(smth_rate)

    return smth_rate


def KDE_behaviour(bins_tuple, covariates, sm_size, L, smooth_modes, dev="cpu"):
    """
    Kernel density estimation of the covariates, with Gaussian kernels.
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

    smth_time = smooth_hist(bin_time[None, ...], sm_filter, smooth_modes, dev=dev)
    smth_time /= smth_time.sum()  # normalize
    return smth_time[0, ...], bin_time


# histogram tuning curves
def geometric_tuning(ori_rate, smth_rate, prob):
    r"""
    Compute coherence and sparsity related to the geometric properties of tuning curves.
    """
    # Pearson r correlation
    units = ori_rate.shape[0]
    coherence = np.empty(units)
    for u in range(units):
        x_1 = ori_rate[u].flatten()
        x_2 = smth_rate[u].flatten()
        stds = np.maximum(1e-10, x_1.std() * x_2.std())
        coherence[u] = np.dot(x_1 - x_1.mean(), x_2 - x_2.mean()) / len(x_1) / stds

    # Computes the sparsity of the tuning
    sparsity = np.empty(units)
    for u in range(units):
        pr_squared = np.maximum(1e-10, (prob * ori_rate[u] ** 2).sum())
        sparsity[u] = 1 - (prob * ori_rate[u]).sum() ** 2 / pr_squared

    return coherence, sparsity


def tuning_overlap(tuning):
    """
    Compute the overlap of tuning curves using the inner product of normalized firing maps [1].

    References:

    [1] `Organization of cell assemblies in the hippocampus` (supplementary),
    Kenneth D. Harris, Jozsef Csicsvari*, Hajime Hirase, George Dragoi & Gyorgy Buzsaki

    """
    g = len(tuning.shape) - 1
    tuning_normalized = tuning.mean(axis=tuple(1 + k for k in range(g)), keepdims=True)
    overlap = np.einsum("i...,j...->ij", tuning_normalized, tuning_normalized)
    return overlap

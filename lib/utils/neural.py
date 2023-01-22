import numpy as np
import scipy.signal as signal

import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln

from ..utils.jax import safe_log


# spike trains
def get_ISIs(timeline, spike_ind, prng_state, minimum=1e-8):
    """
    Converts a binary spike train to ISI w.r.t. timeline

    :param jnp.ndarray timeline: shape (out_dims, time)
    :param jnp.ndarray spike_ind: shape (out_dims, time)
    :return:
        list of ISI arrays
    """
    out_dims, ts = spike_ind.shape
    container = []
    for n in range(out_dims):
        inds = spike_ind[n]
        times = timeline[n, inds]

        if prng_state is not None:
            deqn = jr.uniform(prng_state, shape=(ts,)) * tbin
            times += deqn
            prng_state, _ = jr.split(prng_state)

        container.append(jnp.minimum(jnp.diff(times), minimum))

    return container


def get_lagged_ISIs(spiketrain, lags, dt):
    """
    :param np.ndarray spiketrain: input spike trains of shape (time, neurons)
    :return:
        lagged ISIs of shape (time, neurons, lags)
    """
    T, N = spiketrain.shape

    def step(carry, inputs):
        carry = carry.at[:, 0].add(dt)
        out = carry

        # spike reset
        spike_cond = inputs > 0
        carry = carry.at[:, 1:].set(
            jnp.where(spike_cond[:, None], carry[:, :-1], carry[:, 1:])
        )
        carry = carry.at[:, 0].set(jnp.where(spike_cond, 0.0, carry[:, 0]))
        return carry, out

    init = jnp.ones((N, lags)) * jnp.nan
    xs = spiketrain

    _, lagged_ISIs = lax.scan(step, init, xs)
    return lagged_ISIs


def time_rescale(spikes, intensities, dt, max_intervals):
    """
    rate rescaling, computes the log density on the way

    :param jnp.ndarray spikes: (ts, obs_dims)
    :param jnp.ndarray intensities: (ts, obs_dims)
    :param float dt: time bin size
    """
    def step(carry, inputs):
        t_tilde = carry
        intensity, spike = inputs

        t_tilde += intensity * dt
        t_tilde_ = jnp.where(spike, 0., t_tilde)  # reset after spike
        return (t_tilde_, ll), t_tilde if return_t_tilde else None

    init = jnp.nan*jnp.ones((self.renewal.obs_dims))
    _, t_tildes = lax.scan(step, init=init, xs=(intensities, spikes))
    return log_lik, t_tildes


def train_to_ind(train, allow_duplicate):
    if allow_duplicate:
        duplicate = False
        spike_ind = train.nonzero().flatten()
        bigger = jnp.where(train > 1)[0]
        add_on = (spike_ind,)
        for b in bigger:
            add_on += (b * jnp.ones(int(train[b]) - 1, dtype=int),)

        if len(add_on) > 1:
            duplicate = True
        spike_ind = torch.cat(add_on)
        return torch.sort(spike_ind)[0], duplicate
    else:
        return torch.nonzero(train).flatten(), False


def covariates_at_spikes(spiketimes, behaviour_data):
    """
    Returns tuple of covariate arrays at spiketimes for all neurons
    """
    cov_s = tuple([] for n in behaviour_data)
    units = len(spiketimes)
    for u in range(units):
        for k, cov_t in enumerate(behaviour_data):
            cov_s[k].append(cov_t[spiketimes[u]])

    return cov_s


def bin_data(
    bin_size,
    bin_time,
    spikes,
    track_samples,
    behaviour_data=None,
    average_behav=True,
    binned=False,
):
    """
    Bin the spike train into a given bin size.

    :param int bin_size: desired binning of original time steps into new bin
    :param float bin_time: time step of each original bin or time point
    :param np.array spikes: input spikes in train or index format
    :param int track_samples: number of time steps in the recording
    :param tuple behaviour_data: input behavioural time series
    :param bool average_behav: takes the middle element in bins for behavioural data if False
    :param bool binned: spikes is a spike train if True (trials, neurons, time), otherwise
                        it is a list of spike time indices (trials, neurons)*[spike indices]
    :return:
        tbin, resamples, rc_t, rcov_t
    """
    tbin = bin_size * bin_time
    resamples = int(np.floor(track_samples / bin_size))
    centre = bin_size // 2
    # leave out data with full bins

    rcov_t = ()
    if behaviour_data is not None:
        if isinstance(average_behav, list) is False:
            average_behav = [average_behav for _ in range(len(behaviour_data))]

        for k, cov_t in enumerate(behaviour_data):
            if average_behav[k]:
                rcov_t += (
                    cov_t[: resamples * bin_size].reshape(resamples, bin_size).mean(1),
                )
            else:
                rcov_t += (cov_t[centre : resamples * bin_size : bin_size],)

    if binned:
        rc_t = (
            spikes[:, : resamples * bin_size]
            .reshape(spikes.shape[0], resamples, bin_size)
            .sum(-1)
        )
    else:
        units = len(spikes)
        rc_t = np.zeros((units, resamples))
        for u in range(units):
            retimes = np.floor(spikes[u] / bin_size).astype(int)
            np.add.at(rc_t[u], retimes[retimes < resamples], 1)

    return tbin, resamples, rc_t, rcov_t


def binned_to_indices(spiketrain):
    """
    Converts a binned spike train into spike time indices (with duplicates)

    :param np.array spiketrain: the spike train to convert
    :returns: spike indices denoting spike times in units of time bins
    :rtype: np.array
    """
    spike_ind = spiketrain.nonzero()[0]
    bigger = np.where(spiketrain > 1)[0]
    add_on = (spike_ind,)
    for b in bigger:
        add_on += (b * np.ones(int(spiketrain[b]) - 1, dtype=int),)
    spike_ind = np.concatenate(add_on)
    return np.sort(spike_ind)


def spike_correlogram(
    spiketrain,
    lag_step,
    lag_points,
    segment_len,
    start_step=0,
    ref_point=0,
    cross=True,
    correlation=False,
    dev="cpu",
):
    """
    Get the temporal correlogram of spikes in a given population. Computes

    .. math::
            C_{ij}(\tau) = \langle S_i(t) S_j(t + \tau) \rangle

    or if correlation flag is True, it computes

    .. math::
            C_{ij}(\tau) = \text{Corr}[ S_i(t), S_j(t + \tau) ]

    :param np.array spiketrain: array of population activity of shape (neurons, time)
    :param int lag_range:
    :param int N_period:
    :param
    :param list start_points: list of integers of time stamps where to start computing the correlograms
    :param bool cross: compute the full cross-correlogram over the population, otherwise compute only auto-correlograms
    """
    units = spiketrain.shape[0]
    spikes = jnp.array(spiketrain, device=dev).float()
    spikes_unfold = spikes[
        :, start_step : start_step + (lag_points - 1) * lag_step + segment_len
    ].unfold(
        -1, segment_len, lag_step
    )  # n, t, f

    if cross:
        cg = []
        for u in range(units):
            a = spikes_unfold[u:, ref_point : ref_point + 1, :]
            b = spikes_unfold[u : u + 1, ...]
            if correlation:
                a_m = a.mean(-1, keepdims=True)
                b_m = b.mean(-1, keepdims=True)
                a_std = a.std(-1)
                b_std = b.std(-1)
                cg.append(((a - a_m) * (b - b_m)).mean(-1) / (a_std * b_std))
            else:
                cg.append((a * b).mean(-1))  # neurons-u, lags

        cg = torch.cat(cg, dim=0)
    else:
        a = spikes_unfold[:, ref_point : ref_point + 1, :]
        b = spikes_unfold
        if correlation:
            a_m = a.mean(-1, keepdims=True)
            b_m = b.mean(-1, keepdims=True)
            a_std = a.std(-1)
            b_std = b.std(-1)
            cg = ((a - a_m) * (b - b_m)).mean(-1) / (a_std * b_std)
        else:
            cg = (a * b).mean(-1)  # neurons, lags

    return cg.cpu().numpy()


def compute_ISI_LV(sample_bin, spiketimes):
    r"""
    Compute the local variation measure and the interspike intervals.

    .. math::
            LV = 3 \langle \left( \frac{\Delta_{k-1} - \Delta_{k}}{\Delta_{k-1} + \Delta_{k}} \right)^2 \rangle

    local coefficient of variation LV classification:
        regular spiking for LV ∈ [0, 0.5], irregular
        for LV ∈ [0.5, 1], and bursty spiking for LV > 1.
        S. Shinomoto, K. Shima, and J. Tanji. Differences in spiking patterns
        among cortical neurons. Neural Computation, 15(12):2823–2842, 2003.
        
    References:

    [1] `A measure of local variation of inter-spike intervals',
    Shigeru Shinomoto, Keiji Miura Shinsuke Koyama (2005)
    """
    ISI = []
    LV = []
    units = len(spiketimes)
    for u in range(units):
        if len(spiketimes[u]) < 3:
            ISI.append([])
            LV.append([])
            continue
        ISI_ = (spiketimes[u][1:] - spiketimes[u][:-1]) * sample_bin
        LV.append(3 * (((ISI_[:-1] - ISI_[1:]) / (ISI_[:-1] + ISI_[1:])) ** 2).mean())
        ISI.append(ISI_)

    return ISI, LV



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


def spike_var_MI(rate, prob):
    """
    Mutual information analysis for inhomogeneous Poisson process rate variable.

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



def smooth_hist(rate_binned, sm_filter, bound):
    r"""
    Neurons is the batch dimension, parallelize the convolution for 1D, 2D or 3D
    bound indicates the padding mode ('periodic', 'repeat', 'zeros')
    sm_filter should had odd sizes for its shape

    :param np.array rate_binned: input histogram array of shape (units, ndim_1, ndim_2, ...)
    :param np.array sm_filter: input filter array of shape (ndim_1, ndim_2, ...)
    :param list bound: list of strings (per dimension) to indicate convolution boundary conditions
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


def KDE_behaviour(bins_tuple, covariates, sm_size, L, smooth_modes):
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

    smth_time = smooth_hist(bin_time[None, ...], sm_filter, smooth_modes)
    smth_time /= smth_time.sum()  # normalize
    return smth_time[0, ...], bin_time


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


### sampling ###
def gen_IRP(interval_sampler, rate, dt, samples=100):
    """
    Sample event times from an Inhomogenous Renewal Process with a given rate function
    samples is an algorithm parameter, should be around the expect number of spikes
    Assumes piecewise constant rate function

    Samples intervals from :math:`q(\Delta)`, parallelizes sampling

    :param np.ndarray rate: (trials, neurons, timestep)
    :param ISI_dist interval_dist: renewal interval distribution :math:`q(\tau)`
    :returns: event times as integer indices of the rate array time dimension
    :rtype: list of spike time indices as indexed by rate time dimension
    """
    sim_samples = rate.shape[2]
    N = rate.shape[1]  # neurons
    trials = rate.shape[0]
    T = (
        np.transpose(np.cumsum(rate, axis=-1), (2, 0, 1)) * dt
    )  # actual time to rescaled, (time, trials, neuron)

    psT = 0
    sT_cont = []
    while True:

        sT = psT + np.cumsum(
            interval_sampler(
                (
                    samples,
                    trials,
                )
            ),
            axis=0,
        )
        sT_cont.append(sT)

        if not (T[-1, ...] >= sT[-1, ...]).any():  # all False
            break

        psT = np.tile(sT[-1:, ...], (samples, 1, 1))

    sT_cont = np.stack(sT_cont, axis=0).reshape(-1, trials, N)
    samples_tot = sT_cont.shape[0]
    st = []

    iterator = tqdm(range(samples_tot), leave=False)
    for ss in iterator:  # AR assignment
        comp = np.tile(sT_cont[ss : ss + 1, ...], (sim_samples, 1, 1))
        st.append(np.argmax((comp < T), axis=0))  # convert to rescaled time indices

    st = np.array(st)  # (samples_tot, trials, neurons)
    st_new = []
    for st_ in st.reshape(samples_tot, -1).T:
        if not st_.any():  # all zero
            st_new.append(np.array([]).astype(int))
        else:  # first zeros in this case counts as a spike indices
            for k in range(samples_tot):
                if st_[-1 - k] != 0:
                    break
            if k == 0:
                st_new.append(st_)
            else:
                st_new.append(st_[:-k])

    return st_new  # list of len trials x neurons


def gen_ZIP(prng_state, mean, alpha):
    zero_mask = jr.bernoulli(prng_state, alpha)
    prng_state, _ = jr.split(prng_state)
    cnts = jr.poisson(prng_state, mean)
    return (1.0 - zero_mask) * cnts
    
    
def gen_NB(prng_state, mean, r):
    s = (mean / r) * jr.gamma(
        prng_state, r
    )  # becomes delta around rate*tbin when r to infinity, cap at 1e12
    prng_state, _ = jr.split(prng_state)
    return jr.poisson(prng_state, s)


def gen_CMP(prng_state, mu, nu):#, max_rejections=1000):
    """
    Use rejection sampling to sample from the COM-Poisson count distribution. [1]

    References:

    [1] `Bayesian Inference, Model Selection and Likelihood Estimation using Fast Rejection
         Sampling: The Conway-Maxwell-Poisson Distribution`, Alan Benson, Nial Friel (2021)

    :param numpy.array rate: input rate of shape (..., time)
    :param float tbin: time bin size
    :param float eps: order of magnitude of P(N>1)/P(N<2) per dilated Bernoulli bin
    :param int max_count: maximum number of spike counts per bin possible
    :returns:
        inhomogeneous Poisson process sample (numpy.array)
    """
    def cond_fun(val):
        _, left_inds, _ = val
        return left_inds.any()
        
    def poiss_reject(val):
        Y, left_inds, prng_state = val
        prng_keys = jr.split(prng_state, 3)
        
        mu_ = jnp.floor(mu)
        y_dash = jr.poisson(prng_keys[0], mu)

        log_alpha = (nu - 1) * (
            (y_dash - mu_) * safe_log(mu) - gammaln(y_dash + 1.) + gammaln(mu_ + 1.)
        )

        u = jr.uniform(prng_keys[1], shape=mu.shape)
        alpha = jnp.exp(log_alpha)
        selected = (u <= alpha) * left_inds
        
        Y = jnp.where(selected, y_dash, Y)
        left_inds *= (~selected)
#         if k >= max_rejections:
#             raise ValueError("Maximum rejection steps exceeded")
#         else:
#             k += 1
        return Y, left_inds, prng_keys[2]
    
    def geom_reject(val):
        Y, left_inds, prng_state = val
        prng_keys = jr.split(prng_state, 3)
        
        p = 2 * nu / (2 * mu * nu + 1 + nu)
        u_0 = jr.uniform(prng_keys[0], shape=mu.shape)
        y_dash = jnp.floor(safe_log(u_0) / safe_log(1 - p))
        a = jnp.floor(mu / (1 - p) ** (1 / nu))

        log_alpha = (a - y_dash) * safe_log(1 - p) + nu * (
            (y_dash - a) * safe_log(mu) - gammaln(y_dash + 1.)  + gammaln(a + 1.)
        )

        u = jr.uniform(prng_keys[1], shape=mu.shape)
        alpha = jnp.exp(log_alpha)
        selected = (u <= alpha) * left_inds
        
        Y = jnp.where(selected, y_dash, Y)
        left_inds *= (~selected)        
        return Y, left_inds, prng_keys[2]
    
    prng_states = jr.split(prng_state, 2)
    Y = jnp.empty_like(mu)
    Y, _, _ = lax.while_loop(cond_fun, poiss_reject, init_val=(Y, (nu >= 1), prng_states[0]))
    Y, _, _ = lax.while_loop(cond_fun, geom_reject, init_val=(Y, (nu < 1), prng_states[1]))
    
    return Y
#     trials = mu.shape[0]
#     neurons = mu.shape[1]
#     Y = np.empty(mu.shape)

#     for tr in range(trials):
#         for n in range(neurons):
#             mu_, nu_ = mu[tr, n, :], nu[tr, n, :]
            
#             # Poisson rejection sampling for nu >= 1
#             k = 0
#             left_bins = np.where(nu_ >= 1)[0]
#             while len(left_bins) > 0:
#                 mu__, nu__ = mu_[left_bins], nu_[left_bins]
#                 y_dash = jnp.poisson(jnp.tensor(mu__)).numpy()
#                 _mu_ = np.floor(mu__)
#                 alpha = (
#                     mu__ ** (y_dash - _mu_)
#                     / scsps.factorial(y_dash)
#                     * scsps.factorial(_mu_)
#                 ) ** (nu__ - 1)

#                 u = np.random.rand(*mu__.shape)
#                 selected = u <= alpha
#                 Y[tr, n, left_bins[selected]] = y_dash[selected]
#                 left_bins = left_bins[~selected]
#                 if k >= max_rejections:
#                     raise ValueError("Maximum rejection steps exceeded")
#                 else:
#                     k += 1

#             # geometric rejection sampling for nu < 1
#             k = 0
#             left_bins = np.where(nu_ < 1)[0]
#             while len(left_bins) > 0:
#                 mu__, nu__ = mu_[left_bins], nu_[left_bins]
#                 p = 2 * nu__ / (2 * mu__ * nu__ + 1 + nu__)
#                 u_0 = np.random.rand(*p.shape)

#                 y_dash = np.floor(np.log(u_0) / np.log(1 - p))
#                 a = np.floor(mu__ / (1 - p) ** (1 / nu__))
#                 alpha = (1 - p) ** (a - y_dash) * (
#                     mu__ ** (y_dash - a) / scsps.factorial(y_dash) * scsps.factorial(a)
#                 ) ** nu__

#                 u = np.random.rand(*mu__.shape)
#                 selected = u <= alpha
#                 Y[tr, n, left_bins[selected]] = y_dash[selected]
#                 left_bins = left_bins[~selected]
#                 if k >= max_rejections:
#                     raise ValueError("Maximum rejection steps exceeded")
#                 else:
#                     k += 1

#     return Y

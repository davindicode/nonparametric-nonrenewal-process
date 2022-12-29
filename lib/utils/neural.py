import jax
import jax.numpy as jnp
import jax.random as jr

from jax import lax



# spike trains
def get_lagged_ISIs(spiketrain, lags):
    """
    :param np.ndarray spiketrain: input spike trains of shape (time, neurons)
    :return:
        lagged ISIs of shape (time, neurons, lags)
    """
    T, N = spiketrain.shape

    def step(carry, inputs):
        carry = carry.at[:, 0].add(1.0)
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


def train_to_ind(train, allow_duplicate):
    if allow_duplicate:
        duplicate = False
        spike_ind = train.nonzero().flatten()
        bigger = torch.where(train > 1)[0]
        add_on = (spike_ind,)
        for b in bigger:
            add_on += (
                b * torch.ones(int(train[b]) - 1, device=train.device, dtype=int),
            )

        if len(add_on) > 1:
            duplicate = True
        spike_ind = torch.cat(add_on)
        return torch.sort(spike_ind)[0], duplicate
    else:
        return torch.nonzero(train).flatten(), False

    
    
def ind_to_train(self, ind, timesteps):
    train = torch.zeros((timesteps))
    train[ind] += 1
    return train



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

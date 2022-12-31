import jax
import jax.numpy as jnp
import jax.random as jr

from jax import lax



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
        bigger = jnp.where(train > 1)[0]
        add_on = (spike_ind,)
        for b in bigger:
            add_on += (
                b * jnp.ones(int(train[b]) - 1, dtype=int),
            )

        if len(add_on) > 1:
            duplicate = True
        spike_ind = torch.cat(add_on)
        return torch.sort(spike_ind)[0], duplicate
    else:
        return torch.nonzero(train).flatten(), False

    
    
def ind_to_train(self, ind, timesteps):
    train = jnp.zeros((timesteps))
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



def spike_correlogram(spiketrain, lag_step, lag_points, segment_len, start_step=0, ref_point=0, cross=True, correlation=False, dev='cpu'):
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
    spikes_unfold = spikes[:, start_step:start_step+(lag_points-1)*lag_step+segment_len].unfold(-1, segment_len, lag_step) # n, t, f
    
    if cross:
        cg = []
        for u in range(units):
            a = spikes_unfold[u:, ref_point:ref_point+1, :]
            b = spikes_unfold[u:u+1, ...]
            if correlation:
                a_m = a.mean(-1, keepdims=True)
                b_m = b.mean(-1, keepdims=True)
                a_std = a.std(-1)
                b_std = b.std(-1)
                cg.append(((a-a_m)*(b-b_m)).mean(-1)/(a_std*b_std))
            else:
                cg.append((a*b).mean(-1)) # neurons-u, lags
                
        cg = torch.cat(cg, dim=0)
    else:
        a = spikes_unfold[:, ref_point:ref_point+1, :]
        b = spikes_unfold
        if correlation:
            a_m = a.mean(-1, keepdims=True)
            b_m = b.mean(-1, keepdims=True)
            a_std = a.std(-1)
            b_std = b.std(-1)
            cg = ((a-a_m)*(b-b_m)).mean(-1)/(a_std*b_std)
        else:
            cg = (a*b).mean(-1) # neurons, lags

    return cg.cpu().numpy()



def compute_ISI_LV(sample_bin, spiketimes):
    r"""
    Compute the local variation measure and the interspike intervals.
    
    .. math::
            LV = 3 \langle \left( \frac{\Delta_{k-1} - \Delta_{k}}{\Delta_{k-1} + \Delta_{k}} \right)^2 \rangle
    
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
        ISI_ = (spiketimes[u][1:] - spiketimes[u][:-1])*sample_bin
        LV.append( 3 * (((ISI_[:-1] - ISI_[1:]) / (ISI_[:-1] + ISI_[1:]))**2).mean() )
        ISI.append(ISI_)
        
    return ISI, LV
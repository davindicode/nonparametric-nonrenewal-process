import numpy as np

import jax.numpy as jnp



class Dataset:
    def __init__(self, inputs, targets, batches, batch_axis=0, batch_order=None):
        # batch data into inputs for network
        inputs_batched = np.split(inputs, batches, axis=batch_axis)
        targets_batched = np.split(targets, batches, axis=batch_axis)
        self.batches = batches
        if batch_order is None:
            batch_order = np.arange(self.batches)

        self.inputs = [inputs_batched[b] for b in batch_order]
        self.targets = [targets_batched[b] for b in batch_order]

    def __len__(self):
        return self.batches

    def __iter__(self):
        return iter(zip(self.inputs, self.targets))
    
    
def DataLoader():
    obs_inputs: np.ndarray
        
    def __init__(self, obs_inputs):
        self.obs_inputs = obs_inputs
    
    def load(self):
        data = jnp.array(obs_inputs, dtype=dtype)
        return timestamps, data
    
    
    
def SpikeTrainLoader(DataLoader):
    """
    Loading spike trains (binary arrays)
    """
    def set_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for quick log-likelihood evaluation
        Tensor shapes: self.spikes (neuron_dim, batch_dim)

        tfact is the log of time_bin times the spike count
        """
        if self.allow_duplicate is False and spikes.max() > 1:  # only binary trains
            raise ValueError("Only binary spike trains are accepted in set_Y() here")
        super().set_Y(spikes, batch_info)
        
    def set_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for quick log-likelihood evaluation
        Tensor shapes: self.spikes (neuron_dim, batch_dim)
        
        tfact is the log of time_bin times the spike count
        lfact is the log (spike count)!
        """
        super().set_Y(spikes, batch_info)
        batch_edge, _, _ = self.batch_info
        
        self.lfact = []
        self.tfact = []
        self.totspik = []
        for b in range(self.batches):
            spikes = self.all_spikes[..., batch_edge[b]:batch_edge[b+1]]
            self.totspik.append(spikes.sum(-1))
            self.tfact.append(spikes*torch.log(self.tbin.cpu()))
            self.lfact.append(torch.lgamma(spikes+1.))
        
        
        
def ISILoader(DataLoader):
    """
    Loading inter-spike intervals of some point process time series
    """
    def __init__(self, obs_inputs, tensor_type, allow_duplicate, dequantize):
        super().__init__()
        
    def set_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for quick log-likelihood evaluation
        Tensor shapes: self.act [neuron_dim, batch_dim]
        """
        if self.allow_duplicate is False and spikes.max() > 1:  # only binary trains
            raise ValueError("Only binary spike trains are accepted in set_Y() here")
        super().set_Y(spikes, batch_info)
        batch_edge, _, _ = self.batch_info

        self.spiketimes = []
        self.intervals = torch.empty((self.batches, self.trials, self.neurons))
        self.duplicate = np.empty((self.batches, self.trials, self.neurons), dtype=bool)
        for b in range(self.batches):
            spk = self.all_spikes[..., batch_edge[b] : batch_edge[b + 1]]
            spiketimes = []
            for tr in range(self.trials):
                cont = []
                for k in range(self.neurons):
                    s, self.duplicate[b, tr, k] = self.train_to_ind(spk[tr, k])
                    cont.append(s)
                    self.intervals[b, tr, k] = len(s) - 1
                spiketimes.append(cont)
            self.spiketimes.append(
                spiketimes
            )  # batch list of trial list of spike times list over neurons
            
    def sample_helper(self, h, b, neuron, scale, samples):
        """
        MC estimator for NLL function.

        :param torch.Tensor scale: additional scaling of the rate rescaling to preserve the ISI mean

        :returns: tuple of rates, spikes*log(rates*scale), rescaled ISIs
        :rtype: tuple
        """
        batch_edge, _, _ = self.batch_info
        scale = scale.expand(1, self.F_dims)[
            :, neuron, None
        ]  # rescale to get mean 1 in renewal distribution
        rates = self.f(h) * scale
        spikes = self.all_spikes[:, neuron, batch_edge[b] : batch_edge[b + 1]].to(
            self.dt.device
        )
        # self.spikes[b][:, neuron, self.filter_len-1:]
        if (
            self.trials != 1 and samples > 1 and self.trials < h.shape[0]
        ):  # cannot rely on broadcasting
            spikes = spikes.repeat(
                samples, 1, 1
            )  # trial blocks are preserved, concatenated in first dim

        if (
            self.link_type == "log"
        ):  # bit masking seems faster than integer indexing using spiketimes
            n_l_rates = (spikes * (h + torch.log(scale))).sum(-1)
        else:
            n_l_rates = (spikes * torch.log(rates + 1e-12)).sum(
                -1
            )  # rates include scaling

        spiketimes = [[s.to(self.dt.device) for s in ss] for ss in self.spiketimes[b]]
        rISI = self.rate_rescale(neuron, spiketimes, rates, self.duplicate[b])
        return rates, n_l_rates, rISI
        
        
        
def FIRLoader(DataLoader):
    """
    Taking into account windows for FIR filters
    """
    def __init__(self, obs_inputs):
        super().__init__()
        
    def set_Y(self, spikes, batch_info):
        if len(spikes.shape) == 2:  # add in trial dimension
            spikes = spikes[None, ...]

        in_spikes = spikes[..., self.history_len :]
        self.likelihood.set_Y(
            in_spikes, batch_info
        )  # excludes history part of spike train
        self.likelihood.all_spikes = spikes.type(
            self.likelihood.tensor_type
        )  # overwrite

        _, batch_link, batch_initial = self.likelihood.batch_info
        if any(batch_initial[1:]) or all(batch_link[1:]) is False:
            raise ValueError("Filtered likelihood must take in continuous data")

        self.all_spikes = self.likelihood.all_spikes
        self.batch_info = self.likelihood.batch_info
        self.batches = self.likelihood.batches
        self.trials = self.likelihood.trials
        self.tsteps = self.likelihood.tsteps
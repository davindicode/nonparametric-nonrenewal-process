import jax.numpy as jnp
import numpy as np


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

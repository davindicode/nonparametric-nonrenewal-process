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
        
        
        
def ISILoader(DataLoader):
    """
    Loading inter-spike intervals of some point process time series
    """
    def __init__(self, obs_inputs, tensor_type, allow_duplicate, dequantize):
        super().__init__()
        
        
def FIRLoader(DataLoader):
    """
    Taking into account windows for FIR filters
    """
    def __init__(self, obs_inputs):
        super().__init__()
# GLM filters
class _filter(nn.Module):
    """
    GLM coupling filter base class.
    """

    def __init__(self, filter_len, conv_groups, tensor_type):
        """
        Filter length includes instantaneous part
        """
        super().__init__()
        self.conv_groups = conv_groups
        self.tensor_type = tensor_type
        if filter_len <= 0:
            raise ValueError("Filter length must be bigger than zero")
        self.filter_len = filter_len

    def forward(self):
        """
        Return filter values.
        """
        raise NotImplementedError

    def KL_prior(self, importance_weighted):
        """
        Prior of the filter model.
        """
        return 0

    def constrain(self):
        return
    
    
    


class filtered_input(base._VI_object):
    """
    Stimulus filtering as in GLMs
    """

    def __init__(self, input_series, stimulus_filter, tensor_type=torch.float):
        self.register_buffer("input_series", input_series.type(tensor_type))

        self.add_module("filter", stimulus_filter)
        self.history_len = (
            self.filter.history_len
        )  # history excludes instantaneous part

    def sample(self, b, batch_info, samples, net_input, importance_weighted):
        """ """
        _XZ = self.stimulus_filter(XZ.permute(0, 2, 1))[0].permute(
            0, 2, 1
        )  # ignore filter variance
        KL_prior = self.stimulus_filter.KL_prior()

        return _XZ, KL_prior
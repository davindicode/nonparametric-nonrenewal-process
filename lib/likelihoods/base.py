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
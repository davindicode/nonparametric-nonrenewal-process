import numpy as np


class BatchedTimeSeries:
    """
    Data with loading functionality

    Allows for filtering
    """

    def __init__(
        self, timestamps, covariates, ISIs, observations, batch_size, filter_length=0
    ):
        """
        :param np.ndarray timestamps: (ts,)
        :param np.ndarray covariates: (ts, x_dims)
        :param np.ndarray ISIs: (out_dims, ts, order)
        :param np.ndarray observations: includes filter history (out_dims, ts + filter_length)
        """
        pts = len(timestamps)

        # checks
        if covariates is not None:
            assert covariates.shape[0] == pts
        if ISIs is not None:
            assert ISIs.shape[1] == pts
        assert observations.shape[1] == pts + filter_length

        self.batches = int(np.ceil(pts / batch_size))
        self.batch_size = batch_size
        self.filter_length = filter_length

        self.timestamps = timestamps
        self.covariates = covariates
        self.ISIs = ISIs
        self.observations = observations

    def load_batch(self, batch_index):
        t_inds = slice(
            batch_index * self.batch_size, (batch_index + 1) * self.batch_size
        )
        y_inds = slice(
            batch_index * self.batch_size + self.filter_length,
            (batch_index + 1) * self.batch_size + self.filter_length,
        )

        ts = self.timestamps[t_inds]
        xs = self.covariates[t_inds] if self.covariates is not None else None
        deltas = self.ISIs[:, t_inds] if self.ISIs is not None else None

        ys = self.observations[:, y_inds]
        if self.filter_length > 0:
            filt_inds = slice(
                batch_index * self.batch_size,
                batch_index * self.batch_size + self.filter_length + ys.shape[1] - 1,
            )
            ys_filt = self.observations[
                :, filt_inds
            ]  # leave out last time step (causality)
        else:
            ys_filt = None

        return ts, xs, deltas, ys, ys_filt


class BatchedTrials:
    """
    Subsample over batches
    """

    def __init__(
        self, timestamps, covariates, ISIs, observations, batch_size, filter_length=0
    ):
        """
        :param np.ndarray timestamps: (ts,)
        :param np.ndarray covariates: (ts, x_dims)
        :param np.ndarray ISIs: (out_dims, ts, order)
        :param np.ndarray observations: (out_dims, ts)
        """
        pts = len(timestamps)

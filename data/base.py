import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, interp1d


### base class ###
class _dataset:
    """
    utility functions for data cleaning
    """

    def __init__(self, interp_type):
        self.interp_type = interp_type

    def interpolator(self, t, x, kind=None):
        if kind is None:
            kind = self.interp_type

        if kind == "natural":
            return CubicSpline(t, x, bc_type="natural")
        elif kind == "akima":
            return Akima1DInterpolator(t, x)
        elif kind == "linear":
            return lambda t_: np.interp(t_, t, x)
        elif kind == "nearest":
            return interp1d(t, x, kind="nearest")
        else:
            raise ValueError("Invalid interpolation spline type")

    @staticmethod
    def consecutive_arrays(arr, step=1):
        """
        Finds consecutive subarrays satisfying monotonic stepping with step in array.
        Returns on each array element the island index (starting from 1).

        :param list colors: colors to be included in the colormap
        :param string name: name the colormap
        :returns: figure and axis
        :rtype: tuple
        """
        islands = 1  # next island count
        island_ind = np.zeros(arr.shape)
        on_isl = False
        for k in range(1, arr.shape[0]):
            if arr[k] == arr[k - 1] + step:
                if on_isl is False:
                    island_ind[k - 1] = islands
                    on_isl = True
                island_ind[k] = islands
            elif on_isl is True:
                islands += 1
                on_isl = False

        return island_ind

    @staticmethod
    def true_subarrays(arr):
        """
        Finds consecutive subarrays in booleans array.

        :param list colors: colors to be included in the colormap
        :param string name: name the colormap
        :returns: figure and axis
        :rtype: tuple
        """
        on_isl = False
        island_start_ind = []
        island_size = []
        cnt = 0
        for k in range(len(arr)):
            if arr[k]:
                if on_isl is False:
                    island_start_ind.append(k)
                    island_size.append(1)
                    on_isl = True
                else:
                    island_size[cnt] += 1
            elif on_isl is True:
                cnt += 1
                on_isl = False

        return island_start_ind, island_size

    @staticmethod
    def stitch_nans(series, invalids, angular):
        """
        Interpolate between points with NaNs islands, unless they are the ends
        in place operation
        """
        for ind, size in zip(*invalids):
            dinds = np.arange(size)

            if ind == 0:  # copy
                series[dinds + ind] = series[ind + size]
                continue
            elif ind + size == len(series):
                series[dinds + ind] = series[ind - 1]
                continue

            if angular:  # ensure in [0, 2*pi)
                series = series % (2 * np.pi)

            dseries = series[ind + size] - series[ind - 1]

            if angular:  # interpolate with geodesic distances
                if dseries > np.pi:
                    dseries -= 2 * np.pi
                elif dseries < -np.pi:
                    dseries += 2 * np.pi

            series[dinds + ind] = series[ind - 1] + dseries * (dinds + 1) / (size + 1)

        return series

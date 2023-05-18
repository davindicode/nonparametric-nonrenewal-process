import argparse

import pickle

import sys

import numpy as np

sys.path.append("..")  # access to base

from base import _dataset


sessions_properties = {
    "ec014.468": {"rec_id": "ec014.29", "shanks": 12},
}


class hc_2_and_3(_dataset):
    """
    [hc2] as in [1] and [hc3] as in [2] loading, both use a similar format.

    :param string datadir: the directory where the data files are located
    :param string session_id: the name of the session ID which file names are based on
    :param string savefile: the output file location
    :param int shanks: the number of shanks used to record (shank id is at the end of file name)
    :param bool linear_interpolate: interpolate the behaviour linearly as we upsample, otherwise
                                    we have stepwise constant behaviour after upsampling
    :param float speed_limit: the upper limit of animal speed before it becomes a tracking error

    References:

    [1] Mizuseki K, Sirota A, Pastalkova E, Buzsáki G. (2009):
        `Multi-unit recordings from the rat hippocampus made during open field foraging.'
        http://dx.doi.org/10.6080/K0Z60KZ9

    [2] Mizuseki, K., Sirota, A., Pastalkova, E., Diba, K., Buzsáki, G. (2013)
        `Multiple single unit recordings from different rat hippocampal and entorhinal regions while the animals were performing multiple behavioral tasks.',
        CRCNS.org.
        http://dx.doi.org/10.6080/K09G5JRZ
    """

    def __init__(self, datadir, tbin=0.001, interp_type="akima"):
        super().__init__(interp_type)
        self.datadir = datadir
        self.tbin = tbin

    def load_preprocess_save(self, savefile, session_id, shanks, speed_limit=1e3):
        """
        Preprocess timeseries at 20000 Hz


        rat ID date session maze shanks channels reward duration
        (minutes)
        size
        (GB)
        ec13 2-Feb-06 ec013.527 180 cm 4 31 food 17.7 1.3
        ec13 2-Feb-06 ec013.528 180 cm 4 31 food 26.7 2.0
        ec13 2-Feb-06 ec013.529 180 cm 4 31 water 30.8 2.3
        ec13 10-Feb-06 ec013.713 180 cm 4 31 water 37.8 2.7
        ec13 10-Feb-06 ec013.714 180 cm 4 31 water 38.0 2.7
        ec13 13-Feb-06 ec013.754 180 cm 4 31 water 53.9 3.8
        ec13 13-Feb-06 ec013.755 180 cm 4 31 food 32.8 2.4
        ec13 13-Feb-06 ec013.756 180 cm 4 31 water 40.4 2.8
        ec13 13-Feb-06 ec013.757 180 cm 4 31 water 40.3 2.8
        ec13 16-Feb-06 ec013.808 180 cm 4 31 water 36.8 2.6
        ec13 18-Feb-06 ec013.844 180 cm 4 31 water 30.6 2.2
        ec14 22-Feb-07 ec014.277 180 cm 8 64 water 91.9 15
        ec14 25-Feb-07 ec014.333 180 cm 8 64 water 93.5 13
        ec14 15-Mar-07 ec014.793 120 cm 8 64 water 47.4 6.9
        ec14 15-Mar-07 ec014.800 120 cm 8 64 water 49.3 7.2
        ec14 19-Mar-07 ec015.041 120 cm 8 64 water 47.1 6.8
        ec14 19-Mar-07 ec015.047 120 cm 8 64 water 49.4 7.4
        ec16 19-Sep-07 ec016.397 180 cm 8 55 water 90.8 12
        ec16 21-Sep-07 ec016.430 180 cm 8 55 water 106.8 13
        ec16 22-Sep-07 ec016.448 180 cm 8 55 water 90.8 12
        ec16 1-Oct-07 ec016.582 180 cm 8 56 water 90.9 11
        """
        tbin, datadir = self.tbin, self.datadir

        ### spikes ###
        sample_bin = 1.0 / 20000  # s, time binning of electrophysiological data

        spiketimes, spikeclusters, totclusters = [], [], []
        for k in range(1, shanks + 1):
            f_res = open(datadir + session_id + ".res.{}".format(k), "r").read()
            f_clu = open(datadir + session_id + ".clu.{}".format(k), "r").read()

            clu = np.array(f_clu.split(sep="\n")[:-1]).astype(float)
            totclusters.append(clu[0] - 2)
            # index 0 represents artifacts, 1 – noise/nonclusterable units, 2 and above – isolated units
            clu = clu[1:]
            C0 = np.where(clu == 0)[0]
            C1 = np.where(clu == 1)[0]
            indx = list(C0) + list(C1)

            spikeclusters.append(np.delete(clu, indx) - 2)
            res = np.array(f_res.split(sep="\n")[:-1]).astype(float)
            spiketimes.append(np.delete(res, indx))

        totclusters = np.array(totclusters).astype(int)
        units = totclusters.sum()

        sep_t_spike, track_samples, u = [], 0, 0
        for k in range(shanks):
            for l in range(
                totclusters[k]
            ):  # filter based on cluster and resample to tbin
                arr = np.rint(
                    np.sort(spiketimes[k][spikeclusters[k] == l]) * sample_bin / tbin
                ).astype(int)
                if len(arr) < 2:  # silent neurons
                    units -= 1
                    print(
                        "Less than 2 spikes on channel shank {} cluster {}, excluded from data.".format(
                            k, l
                        )
                    )
                    continue

                u += 1
                sep_t_spike.append(arr)
                if track_samples < arr[-1]:
                    track_samples = arr[-1]
            k += 1

        time_glob = np.arange(track_samples) * tbin

        # ISI statistics
        ISI = []
        for u in range(units):
            ISI.append((sep_t_spike[u][1:] - sep_t_spike[u][:-1]) * tbin * 1000)  # ms

        refract_viol = np.empty((units))
        viol_ISI = 2.0  # s
        for u in range(units):
            refract_viol[u] = (ISI[u] <= viol_ISI).sum() / len(ISI[u])

        neural = {
            "spike_time_inds": sep_t_spike,
            "refract_viol": refract_viol,
        }

        ### behaviour ###
        behav_tbin = 1.0 / 39.06  # s

        f_pos = open(datadir + session_id + ".whl", "r").read()
        l = f_pos.split(sep="\n")[:-1]
        pos = np.array([i.split("\t") for i in l]).astype(float)
        pos_1 = pos[:, :2]
        pos_2 = pos[:, 2:]

        dp = pos_1 - pos_2
        hd_beh = np.unwrap(np.angle(dp[:, 0] + dp[:, 1] * 1j))
        x_beh = (pos_1[:, 0] + pos_2[:, 0]) / 2.0
        y_beh = (pos_1[:, 1] + pos_2[:, 1]) / 2.0

        invalids = self.true_subarrays(
            (pos_1[:, 0] == -1.0)
            | (pos_1[:, 1] == -1.0)
            | (pos_2[:, 0] == -1.0)
            | (pos_2[:, 1] == -1.0)
        )
        # invalids = self.true_subarrays(xy_nan)
        x_beh = self.stitch_nans(x_beh, invalids, angular=False)
        y_beh = self.stitch_nans(y_beh, invalids, angular=False)

        # empirical velocities used to smoothen sudden jumps
        vx_beh = (x_beh[1:] - x_beh[:-1]) / behav_tbin  # mm/s
        vy_beh = (y_beh[1:] - y_beh[:-1]) / behav_tbin  # mm/s
        s_beh = np.sqrt(vx_beh**2 + vy_beh**2)
        s_beh[(x_beh[1:] == -1) | (x_beh[:-1] == -1)] = -1  # invalid

        inval_ind_, inval_size_ = self.true_subarrays(
            s_beh == -1
        )  # at edges, ignore these time points
        if (
            len(inval_size_) == 2
        ):  # locations at which this or next time step is invalid

            inds = (
                np.where(s_beh[inval_size[0] : inval_ind[-1]] > speed_limit)[0]
                + inval_size_[0]
            )  # potential tracking errors

            assert len(inds) % 2 == 0  # pairs indicating segments
            for k in range(len(inds) // 2):
                x_prev = inds[2 * k]
                x_next = inds[2 * k + 1] + 1
                steps = x_next - x_prev

                # linear interpolation
                t = np.arange(1, steps)

                dx = (x_beh[x_next] - x_beh[x_prev]) / steps
                x_beh[inds[2 * k] + 1 : inds[2 * k + 1] + 1] = x_beh[x_prev] + dx * t

                dy = (y_beh[x_next] - y_beh[x_prev]) / steps
                y_beh[inds[2 * k] + 1 : inds[2 * k + 1] + 1] = y_beh[x_prev] + dy * t

        # resample behaviour from 39.06 Hz to 20000 Hz
        time_behav = behav_tbin * np.arange(len(x_beh))

        # interpolator for invalid data
        x_t = self.interpolator(time_behav, x_beh)(time_glob)
        y_t = self.interpolator(time_behav, y_beh)(time_glob)
        hd_t = self.interpolator(time_behav, np.unwrap(hd_beh))(time_glob)

        # eeg at 1250 Hz
        eeg_1250Hz = np.fromfile(
            open(datadir + session_id + ".eeg", "rb"), dtype=np.int16
        )
        eeg_channels = int(np.rint(len(eeg_1250Hz) / (track_samples * tbin * 1250)))
        print("EEG channels:", eeg_channels)
        eeg_1250Hz = np.reshape(eeg_1250Hz, (-1, eeg_channels)).T  # (channel, time)

        time_1250Hz = np.arange(eeg_1250Hz.shape[1]) / 1250
        eeg_t = []
        for c in range(eeg_channels):  # resample
            eeg_t.append(self.interpolator(time_1250Hz, eeg_1250Hz[c])(time_glob))
        eeg_t = np.stack(eeg_t)

        covariates = {
            "x": x_t,
            "y": y_t,
            "hd": hd_t,
            "eeg": eeg_t,
        }

        # export
        data = {
            "sample_bin": tbin,
            "ts": time_glob,
            "covariates": covariates,
            "neural": neural,
        }

        if savefile is None:
            return data
        else:
            pickle.dump(data, open(savefile, "wb"), pickle.HIGHEST_PROTOCOL)


def main():
    ## parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Preprocess hc-2 and hc-3 datasets",
    )
    subparsers = parser.add_subparsers(dest="datatype")

    parser = subparsers.add_parser(
        "hc3_linear", help="Fit model to hc3 linear track datasets."
    )

    parser.add_argument("--session_id", type=str)
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--savedir", type=str)

    args = parser.parse_args()

    session_id = args.session_id
    save_dir = args.savedir
    data_dir = args.datadir

    ### processing ###
    shanks = sessions_properties[session_id]["shanks"]
    rec_id = sessions_properties[session_id]["rec_id"]

    datadir = data_dir + rec_id + "/" + session_id + "/"
    savefile = save_dir + "{}.p".format(rec_id + "_" + session_id)

    data_class = hc_2_and_3(datadir)
    data_class.load_preprocess_save(savefile, session_id, shanks)


if __name__ == "__main__":
    main()

import numpy as np


# tools
def filter_signal(signal, f_min, f_max, sample_bin):
    """
    Filter in Fourier space by multiplying with a box function for (f_min, f_max).
    """
    track_samples = signal.shape[0]
    Df = 1 / sample_bin / track_samples
    low_ind = np.floor(f_min / Df).astype(int)
    high_ind = np.ceil(f_max / Df).astype(int)

    g_fft = np.fft.rfft(signal)
    mask = np.zeros_like(g_fft)
    mask[low_ind:high_ind] = 1.0
    g_fft *= mask
    signal_ = np.fft.irfft(g_fft)

    if track_samples % 2 == 1:  # odd
        signal_ = np.concatenate((signal_, signal_[-1:]))

    return signal_


def find_peaks(signal, min_node=True, max_node=True):
    """
    node and anti-nodes
    """
    T = []
    for t in range(1, signal.shape[0] - 1):
        if signal[t - 1] < signal[t] and signal[t + 1] < signal[t] and max_node:
            T.append(t)
        elif signal[t - 1] > signal[t] and signal[t + 1] > signal[t] and min_node:
            T.append(t)

    return T


def linear_theta_phase(eeg_filt):
    """
    find LFP peaks
    """
    T = find_peaks(eeg_filt)
    print("Total peaks:", len(T))

    # remove peaks that are too close (signal artifacts)
    dT = np.array(T[1:]) - np.array(T[:-1])
    rem_inds = np.where(dT < 20)[0]
    for rem_ind in rem_inds[::-1]:  # traverse backward while removing
        T.remove(T[rem_ind])

    # compute linear phase
    min_time = np.zeros(eeg_filt.shape[0])
    period_time = np.zeros(eeg_filt.shape[0])
    baseline_time = np.zeros(eeg_filt.shape[0])
    prev = -1
    T.append(2 * T[len(T) - 1] - T[len(T) - 2])
    for t in range(min_time.shape[0]):
        if t == T[prev + 1]:
            prev += 1

        if prev > -1:
            min_time[t] = T[prev]
            if prev + 1 < len(T):
                period_time[t] = T[prev + 1] - T[prev]
            else:  # use previous period as estimate
                period_time[t] = T[prev] - T[prev - 1]
        else:
            min_time[t] = 2 * T[0] - T[1]
            period_time[t] = T[1] - T[0]

        baseline_time[t] = prev + 1

    timeline = np.arange(eeg_filt.shape[0]) - min_time
    lin_t = (timeline / period_time + baseline_time) * np.pi
    return lin_t


def get_L_R_run(
    x_t, ini_class, margin=10.0, base_low_thresh=75.0, base_high_thresh=275.0
):
    """
    Classify runs
    """
    low_thresh = base_low_thresh
    high_thresh = base_high_thresh
    class_x_t = np.zeros_like(x_t)

    class_names = {
        0: "L-turn",
        1: "L-to-R",
        2: "R-to-L",
        3: "R-turn",
    }
    inv_map = {v: k for k, v in class_names.items()}

    prev_class = inv_map[ini_class]
    for i, x in enumerate(x_t):
        if x > high_thresh:
            new_class = inv_map["R-turn"]
            high_thresh = base_high_thresh - margin

        elif x < low_thresh:
            new_class = inv_map["L-turn"]
            low_thresh = base_low_thresh + margin

        else:
            if prev_class == inv_map["L-to-R"] or prev_class == inv_map["R-to-L"]:
                new_class = prev_class

            elif prev_class == inv_map["L-turn"]:
                new_class = inv_map["L-to-R"]
                low_thresh = base_low_thresh - margin
                high_thresh = base_high_thresh

            elif prev_class == inv_map["R-turn"]:
                new_class = inv_map["R-to-L"]
                high_thresh = base_high_thresh + margin
                low_thresh = base_low_thresh

        class_x_t[i] = new_class
        prev_class = new_class

    return class_x_t, class_names

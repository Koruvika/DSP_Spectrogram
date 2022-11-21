import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import stats
import statistics
import thinkdsp


def zero_crossing_rate(frame):
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(count_zero)


def energy(frame):
    return np.sum(frame ** 2)


def sta(vu_frame):
    count = len(vu_frame)
    return np.sum(vu_frame) / np.float64(count)


def lf(frame, sampling_rate):
    # Create wave proviced amplitude at given time and sampling rate
    temp_sig = thinkdsp.Wave(ys=frame, framerate=sampling_rate)
    sp = temp_sig.make_spectrum()
    # Remove frequency component that have frequency value > 1000Hz
    sp.low_pass(1000)
    filter_wave = sp.make_wave()
    return energy(filter_wave.ys)/energy(frame)


def feature_extraction(signal, sampling_rate, window, step):
    window = int(window)
    step = int(step)

    number_of_samples = len(signal)  # total number of samples
    current_position = 0
    count_fr = 0

    # define list of feature names
    feature_names = ["zcr", "energy", "lf"]

    features = []
    # for each short-term window to end of signal
    while current_position + window - 1 < number_of_samples:
        count_fr += 1
        # get current window
        x = signal[current_position:current_position + window]
        # update window position
        current_position = current_position + step

        feature_vector = np.zeros((3, 1))

        # zero crossing rate
        feature_vector[0] = zero_crossing_rate(x)
        # short-term energy
        feature_vector[1] = energy(x)
        # Low-band over Full-band
        feature_vector[2] = lf(x, sampling_rate)

        features.append(feature_vector)

    features = np.concatenate(features, 1)

    return features, feature_names


def histogram_based_method(st_energy, user_defined_value, pos_1, pos_2):
    w = user_defined_value
    y, x = np.histogram(st_energy, bins=len(st_energy))

    indexes = sig.find_peaks(y)[0]

    max1 = x[indexes[pos_1]]
    max2 = x[indexes[pos_2]]
    T = np.float64(w * max1 + max2) / np.float64(w + 1)
    return T


def segment_limits(data, threshold, st_step, last_frames):
    max_indices = np.where(data < threshold)[0]

    # get the indices of the frames that satisfy the thresholding
    index = 0
    seg_limits = []
    time_clusters = []

    while index < len(max_indices):
        # for each of the detected onset indices
        cur_cluster = [max_indices[index]]
        if index == len(max_indices)-1:
            cur_cluster.append(last_frames)
            break
        while max_indices[index+1] - cur_cluster[-1] <= 2:
            cur_cluster.append(max_indices[index+1])
            index += 1
            if abs(max_indices[index] - last_frames) <= 2:
                cur_cluster.append(last_frames)
            if index == len(max_indices)-1:
                break
        index += 1
        time_clusters.append(cur_cluster)
        seg_limits.append([cur_cluster[0] * st_step,
                           cur_cluster[-1] * st_step])
    return seg_limits


def silence_discrimination(signal, sampling_rate, st_win=0.020, st_step=0.020):
    # Feature extraction
    st_feats, _ = feature_extraction(signal, sampling_rate,
                                     st_win * sampling_rate,
                                     st_step * sampling_rate)

    st_energy = st_feats[1, :]

    ste_threshold = histogram_based_method(st_energy[:], 0.3, 0, 1)

    seg_limits = segment_limits(st_energy[:], ste_threshold, st_step,
                                round(np.float64(len(signal)) / (st_win * sampling_rate)))
    # Remove very small segments:
    min_duration = 0.015
    seg_limits_2 = []
    for s_lim in seg_limits:
        if s_lim[1] - s_lim[0] > min_duration:
            seg_limits_2.append(s_lim)
    seg_limits = seg_limits_2
    return seg_limits

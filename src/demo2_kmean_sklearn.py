import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance

from SpeechSegment import silence_discrimination
from scipy.io.wavfile import read
from scipy import fft
import scipy.signal as sig
import thinkdsp
import os
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

n_clusters = 5
f = open("result.txt", "a")


def get_center_vowel(wave):
    n, a = 9, 1
    b = [1.0 / n] * n
    yy = sig.lfilter(b, a, wave.ys)
    seg_limits = silence_discrimination(yy, wave.framerate, 0.020, 0.020)
    length_center = seg_limits[-1][0] - seg_limits[0][1]
    start = seg_limits[0][1] + length_center/3
    end = seg_limits[0][1] + 2*length_center/3
    return start, end


def get_segment_audio(audio, fs, start, end):
    start_frame = int(start * fs)
    end_frame = int(end * fs)
    segment_audio = audio[start_frame:end_frame+1]
    return segment_audio


def get_fft_of_segment_audio(segment_audio, fs, n_size=1024):
    N_SAMPLE_ON_10MS = int(fs * 0.01)
    N_SAMPLE_ON_20MS = int(fs * 0.02)
    N_SAMPLE_ON_30MS = int(fs * 0.03)

    n = segment_audio.shape[0]

    s = 0
    e = N_SAMPLE_ON_30MS

    ffts = []
    while e < n:
        feature = fft.fft(segment_audio[s: e], n_size)
        feature = np.abs(feature)
        feature = feature / np.linalg.norm(feature)
        ffts.append(feature)
        s += N_SAMPLE_ON_20MS
        e += N_SAMPLE_ON_20MS

    feature = np.mean(ffts, axis=0)
    return feature


def get_vowel_predict(index):
    if index == 0:
        return "u"
    if index == 1:
        return "e"
    if index == 2:
        return "o"
    if index == 3:
        return "a"
    if index == 4:
        return "i"


def calculate_distance_fft(feature1, feature2):
    return np.sum(np.abs(feature1 - feature2), axis=0) / feature2.shape[0]


def inference(testfile, u_center, e_center, o_center, a_center, i_center):
    wave = thinkdsp.read_wave(filename=testfile)
    start, end = get_center_vowel(wave)
    segment_audio = get_segment_audio(wave.ys, wave.framerate, start, end)
    feature = get_fft_of_segment_audio(segment_audio, wave.framerate)
    a_distance = np.mean(np.linalg.norm(feature - a_center, axis=1))
    u_distance = np.mean(np.linalg.norm(feature - u_center, axis=1))
    i_distance = np.mean(np.linalg.norm(feature - i_center, axis=1))
    e_distance = np.mean(np.linalg.norm(feature - e_center, axis=1))
    o_distance = np.mean(np.linalg.norm(feature - o_center, axis=1))

    # u: 0, e: 1, o: 2, a: 3, i: 4
    distance = np.array([u_distance, e_distance, o_distance, a_distance, i_distance])

    predict = np.argmin(distance)

    label = testfile.split(".")[-2][-1]

    f.write(f"{testfile} {get_vowel_predict(predict)} {label}\n")

    return get_vowel_predict(predict), label


if __name__ == "__main__":
    vowel_files = []

    folders = os.listdir("../data/train")
    for folder in folders:
        files = os.listdir("../data/train/" + folder)
        for file in files:
            vowel_files.append("../data/train/" + folder + "/" + file)

    ffts = []
    labels = []
    for file in vowel_files:
        wave = thinkdsp.read_wave(filename=file)
        start, end = get_center_vowel(wave)
        segment_audio = get_segment_audio(wave.ys, wave.framerate, start, end)
        feature = get_fft_of_segment_audio(segment_audio, wave.framerate)
        check = np.sum(np.isnan(feature))
        ffts.append(feature)
        labels.append(file.split(".")[-2][-1])

    ffts = np.array(ffts)
    u_z = []
    e_z = []
    o_z = []
    a_z = []
    i_z = []

    for i in range(len(ffts)):
        if labels[i] == "u":
            u_z.append(ffts[i])
            continue
        if labels[i] == "e":
            e_z.append(ffts[i])
            continue
        if labels[i] == "o":
            o_z.append(ffts[i])
            continue
        if labels[i] == "a":
            a_z.append(ffts[i])
            continue
        if labels[i] == "i":
            i_z.append(ffts[i])
            continue
    u_z = np.array(u_z)
    e_z = np.array(e_z)
    o_z = np.array(o_z)
    a_z = np.array(a_z)
    i_z = np.array(i_z)

    kmeans_u = KMeans(n_clusters=n_clusters, random_state=0).fit(u_z)
    kmeans_e = KMeans(n_clusters=n_clusters, random_state=0).fit(e_z)
    kmeans_o = KMeans(n_clusters=n_clusters, random_state=0).fit(o_z)
    kmeans_a = KMeans(n_clusters=n_clusters, random_state=0).fit(a_z)
    kmeans_i = KMeans(n_clusters=n_clusters, random_state=0).fit(i_z)

    vowel_files = []
    folders = os.listdir("../data/test")
    for folder in folders:
        files = os.listdir("../data/test/" + folder)
        for file in files:
            vowel_files.append("../data/test/" + folder + "/" + file)

    # predict
    predict_results = []
    label_results = []
    for file in vowel_files:
        predict, label = inference(file, kmeans_u.cluster_centers_, kmeans_e.cluster_centers_, kmeans_o.cluster_centers_, kmeans_a.cluster_centers_, kmeans_i.cluster_centers_)
        predict_results.append(predict)
        label_results.append(label)

    # show confusion matrix
    conf = confusion_matrix(label_results, predict_results, labels=["u", "e", "o", "a", "i"])
    alphabets = ['u', 'e', 'o', 'a', 'i']
    figure = plt.figure()
    axes = figure.add_subplot(111)
    caxes = axes.matshow(conf, interpolation='nearest', cmap="seismic")
    figure.colorbar(caxes)
    for (i, j), z in np.ndenumerate(conf):
        axes.text(j, i, '{:d}'.format(z), ha='center', va='center')
    axes.set_xticklabels([''] + alphabets)
    axes.set_yticklabels([''] + alphabets)
    acc = 0
    for i in range(len(conf)):
        acc += conf[i, i]
    acc /= len(vowel_files)
    axes.text(0, 5, f"Accuracy: {acc}")
    plt.show()

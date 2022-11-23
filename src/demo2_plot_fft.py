import numpy as np
import os
from numpy.random import RandomState
from sklearn.manifold import TSNE
import thinkdsp
from demo2_kmean_sklearn import get_segment_audio, get_fft_of_segment_audio, get_center_vowel
import matplotlib.pyplot as plt


def get_number_vowel_predict(index):
    if index == "u":
        return 0
    if index == "e":
        return 1
    if index == "o":
        return 2
    if index == "a":
        return 3
    if index == "i":
        return 4


if __name__ == "__main__":
    rng = RandomState(0)
    plt.style.use("seaborn")

    # read all files
    vowel_files = []
    folders = os.listdir("../data/train")
    for folder in folders:
        files = os.listdir("../data/train/" + folder)
        for file in files:
            vowel_files.append("../data/train/" + folder + "/" + file)

    folders = os.listdir("../data/test")
    for folder in folders:
        files = os.listdir("../data/test/" + folder)
        for file in files:
            vowel_files.append("../data/test/" + folder + "/" + file)

    # get fft and label
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
    number_labels = [get_number_vowel_predict(label) for label in labels]
    number_labels = np.array(number_labels)

    # run tsne
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(ffts)

    u_z = []
    e_z = []
    o_z = []
    a_z = []
    i_z = []

    for i in range(len(z)):
        if labels[i] == "u":
            u_z.append(z[i, :])
            continue
        if labels[i] == "e":
            e_z.append(z[i, :])
            continue
        if labels[i] == "o":
            o_z.append(z[i, :])
            continue
        if labels[i] == "a":
            a_z.append(z[i, :])
            continue
        if labels[i] == "i":
            i_z.append(z[i, :])
            continue
    u_z = np.array(u_z)
    e_z = np.array(e_z)
    o_z = np.array(o_z)
    a_z = np.array(a_z)
    i_z = np.array(i_z)
    plt.scatter(u_z[:, 0], u_z[:, 1], c="red", label="u")
    plt.scatter(e_z[:, 0], e_z[:, 1], c="blue", label="e")
    plt.scatter(o_z[:, 0], o_z[:, 1], c="yellow", label="o")
    plt.scatter(a_z[:, 0], a_z[:, 1], c="pink", label="a")
    plt.scatter(i_z[:, 0], i_z[:, 1], c="green", label="i")
    plt.legend()
    plt.show()


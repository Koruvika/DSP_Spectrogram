import matplotlib.pyplot as plt
import numpy as np

from SpeechSegment import silence_discrimination
from scipy.io.wavfile import read
from scipy import fft
import scipy.signal as sig
import thinkdsp
import os
from sklearn.metrics import confusion_matrix
# import seaborn as sn

# read training fft
a_fft = np.load("a_fft.npy")
u_fft = np.load("u_fft.npy")
e_fft = np.load("e_fft.npy")
i_fft = np.load("i_fft.npy")
o_fft = np.load("o_fft.npy")

f = open("result256.txt", "a")


def get_center_vowel(wave, file):
    n, a = 9, 1
    b = [1.0 / n] * n
    yy = sig.lfilter(b, a, wave.ys)
    seg_limits = silence_discrimination(yy, wave.framerate, 0.020, 0.020)

    length_center = seg_limits[-1][0] - seg_limits[0][1]
    start = seg_limits[0][1] + length_center/4
    end = seg_limits[0][1] + 3*length_center/4
    # if len(seg_limits) != 2:
    #     print(file)
    #     print(seg_limits)
    #     print(length_center)
    #     print(start, end)
    return start, end


def get_segment_audio(audio, fs, start, end):
    start_frame = int(start * fs)
    end_frame = int(end * fs)
    segment_audio = audio[start_frame:end_frame+1]
    return segment_audio


def get_fft_of_segment_audio(segment_audio, fs, n_size=256):
    N_SAMPLE_ON_10MS = int(fs * 0.01)
    N_SAMPLE_ON_20MS = int(fs * 0.02)
    N_SAMPLE_ON_30MS = int(fs * 0.03)

    n = segment_audio.shape[0]
    k = int(n / N_SAMPLE_ON_20MS)

    s = 0
    e = N_SAMPLE_ON_30MS

    ffts = []
    while e < n:
        feature = fft.fft(segment_audio[s: e], n_size)
        ffts.append(feature)
        s += N_SAMPLE_ON_20MS
        e += N_SAMPLE_ON_20MS

    # ffts = []
    # for i in range(k):
    #     s = i * N_FRAME_ON_20MS
    #     e = (i + 1) * N_FRAME_ON_20MS
    #     feature = fft.fft(segment_audio[s: e], n_size)
    #     ffts.append(feature)
    feature = np.mean(ffts, axis=0)
    return feature


def save_fft_of_train_data():
    vowel_files = []

    folders = os.listdir("../data/train")
    for folder in folders:
        files = os.listdir("../data/train/" + folder)
        for file in files:
            vowel_files.append("../data/train/" + folder + "/" + file)

    a_files = [file for file in vowel_files if file.split(".")[-2][-1] == "a"]
    u_files = [file for file in vowel_files if file.split(".")[-2][-1] == "u"]
    e_files = [file for file in vowel_files if file.split(".")[-2][-1] == "e"]
    o_files = [file for file in vowel_files if file.split(".")[-2][-1] == "o"]
    i_files = [file for file in vowel_files if file.split(".")[-2][-1] == "i"]

    a_ffts = []
    for file in a_files:
        wave = thinkdsp.read_wave(filename=file)
        fs, audio = read(file)
        start, end = get_center_vowel(wave, file)
        segment_audio = get_segment_audio(audio, fs, start, end)
        feature = get_fft_of_segment_audio(segment_audio, fs)
        check = np.sum(np.isnan(feature))
        if check >= 1:
            print(file)
        a_ffts.append(feature)
    a_fft = np.mean(a_ffts, axis=0)

    u_ffts = []
    for file in u_files:
        wave = thinkdsp.read_wave(filename=file)
        fs, audio = read(file)
        start, end = get_center_vowel(wave, file)
        segment_audio = get_segment_audio(audio, fs, start, end)
        feature = get_fft_of_segment_audio(segment_audio, fs)
        check = np.sum(np.isnan(feature))
        if check >= 1:
            print(file)
        u_ffts.append(feature)
    u_fft = np.mean(u_ffts, axis=0)

    e_ffts = []
    for file in e_files:
        wave = thinkdsp.read_wave(filename=file)
        fs, audio = read(file)
        start, end = get_center_vowel(wave, file)
        segment_audio = get_segment_audio(audio, fs, start, end)
        feature = get_fft_of_segment_audio(segment_audio, fs)
        check = np.sum(np.isnan(feature))
        if check >= 1:
            print(file)
        e_ffts.append(feature)
    e_fft = np.mean(e_ffts, axis=0)

    o_ffts = []
    for file in o_files:
        wave = thinkdsp.read_wave(filename=file)
        fs, audio = read(file)
        start, end = get_center_vowel(wave, file)
        segment_audio = get_segment_audio(audio, fs, start, end)
        feature = get_fft_of_segment_audio(segment_audio, fs)
        check = np.sum(np.isnan(feature))
        if check >= 1:
            print(file)
        o_ffts.append(feature)
    o_fft = np.mean(o_ffts, axis=0)

    i_ffts = []
    for file in i_files:
        wave = thinkdsp.read_wave(filename=file)
        fs, audio = read(file)
        start, end = get_center_vowel(wave, file)
        segment_audio = get_segment_audio(audio, fs, start, end)
        feature = get_fft_of_segment_audio(segment_audio, fs)
        check = np.sum(np.isnan(feature))
        if check >= 1:
            print(file)
        i_ffts.append(feature)
    i_fft = np.mean(i_ffts, axis=0)

    print(u_fft.shape)
    print(e_fft.shape)
    print(o_fft.shape)
    print(a_fft.shape)
    print(i_fft.shape)

    with open("u_fft.npy", "wb") as f:
        np.save(f, u_fft)
    with open("e_fft.npy", "wb") as f:
        np.save(f, e_fft)
    with open("o_fft.npy", "wb") as f:
        np.save(f, o_fft)
    with open("a_fft.npy", "wb") as f:
        np.save(f, a_fft)
    with open("i_fft.npy", "wb") as f:
        np.save(f, i_fft)

    return u_fft, e_fft, o_fft, a_fft, i_fft


def calculate_distance_fft(feature1, feature2):
    return np.sum(np.abs(feature1 - feature2)) / feature2.shape[0]


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


def inference(testfile):
    wave = thinkdsp.read_wave(filename=testfile)
    fs, audio = read(testfile)
    start, end = get_center_vowel(wave, testfile)
    segment_audio = get_segment_audio(audio, fs, start, end)
    feature = get_fft_of_segment_audio(segment_audio, fs)

    a_distance = calculate_distance_fft(feature, a_fft)
    u_distance = calculate_distance_fft(feature, u_fft)
    i_distance = calculate_distance_fft(feature, i_fft)
    e_distance = calculate_distance_fft(feature, e_fft)
    o_distance = calculate_distance_fft(feature, o_fft)

    # u: 0, e: 1, o: 2, a: 3, i: 4
    distance = np.array([u_distance, e_distance, o_distance, a_distance, i_distance])

    predict = np.argmin(distance)

    label = testfile.split(".")[-2][-1]

    f.write(f"{testfile} {get_vowel_predict(predict)} {label}\n")

    return get_vowel_predict(predict), label


if __name__ == "__main__":
    # u_fft, e_fft, o_fft, a_fft, i_fft = save_fft_of_train_data()

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
        predict, label = inference(file)
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
    plt.show()

    f.close()
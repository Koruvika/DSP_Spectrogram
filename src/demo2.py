import matplotlib.pyplot as plt
import numpy as np
import scipy.fft

from SpeechSegment import silence_discrimination
from scipy.io.wavfile import read
from scipy import fft
import scipy.signal as sig
import thinkdsp
import os


def get_center_vowel(wave):
    n, a = 9, 1
    b = [1.0 / n] * n
    yy = sig.lfilter(b, a, wave.ys)
    seg_limits = silence_discrimination(yy, wave.framerate, 0.020, 0.020)
    length_center = seg_limits[1][0] - seg_limits[0][1]
    start = seg_limits[1][0] + length_center/3
    end = seg_limits[1][0] + 2*length_center/3
    return start, end


def get_segment_audio(audio, fs, start, end):
    start_frame = int(start * fs)
    end_frame = int(end * fs)
    segment_audio = audio[start_frame:end_frame+1]
    return segment_audio


def get_fft_of_segment_audio(segment_audio, fs, n_size=512):
    N_FRAME_ON_10MS = int(fs * 0.01)
    N_FRAME_ON_20MS = int(fs * 0.02)
    N_FRAME_ON_30MS = int(fs * 0.03)

    n = segment_audio.shape[0]
    k = int(n / N_FRAME_ON_20MS)

    ffts = []
    for i in range(k):
        s = i * N_FRAME_ON_20MS
        e = (i + 1) * N_FRAME_ON_20MS
        feature = fft.fft(segment_audio[s: e], n_size)
        ffts.append(feature)
    feature = np.mean(ffts, axis=0)
    return feature


if __name__ == "__main__":
    # wave = thinkdsp.read_wave(filename="../data/01MDA/a.wav")
    # fs, audio = read("../data/01MDA/a.wav")
    # start, end = get_center_vowel(wave)
    # segment_audio = get_segment_audio(audio, fs, start, end)
    # feature = get_fft_of_segment_audio(segment_audio, fs, n_size=512)

    vowel_files = []

    folders = os.listdir("../data")[1:-1]
    for folder in folders:
        files = os.listdir("../data/" + folder)
        for file in files:
            vowel_files.append("../data/" + folder + "/" + file)

    a_files = [file for file in vowel_files if file.split(".")[-2][-1] == "a"]
    u_files = [file for file in vowel_files if file.split(".")[-2][-1] == "u"]
    e_files = [file for file in vowel_files if file.split(".")[-2][-1] == "e"]
    o_files = [file for file in vowel_files if file.split(".")[-2][-1] == "o"]
    i_files = [file for file in vowel_files if file.split(".")[-2][-1] == "i"]

    a_ffts = []
    for file in a_files:
        wave = thinkdsp.read_wave(filename="../data/01MDA/a.wav")
        fs, audio = read("../data/01MDA/a.wav")
        start, end = get_center_vowel(wave)
        segment_audio = get_segment_audio(audio, fs, start, end)
        feature = get_fft_of_segment_audio(segment_audio, fs, n_size=512)
        a_ffts.append(feature)
    a_fft = np.mean(a_ffts, axis=0)

    u_ffts = []
    for file in a_files:
        wave = thinkdsp.read_wave(filename="../data/01MDA/a.wav")
        fs, audio = read("../data/01MDA/a.wav")
        start, end = get_center_vowel(wave)
        segment_audio = get_segment_audio(audio, fs, start, end)
        feature = get_fft_of_segment_audio(segment_audio, fs, n_size=512)
        u_ffts.append(feature)
    u_fft = np.mean(u_ffts, axis=0)

    e_ffts = []
    for file in a_files:
        wave = thinkdsp.read_wave(filename="../data/01MDA/a.wav")
        fs, audio = read("../data/01MDA/a.wav")
        start, end = get_center_vowel(wave)
        segment_audio = get_segment_audio(audio, fs, start, end)
        feature = get_fft_of_segment_audio(segment_audio, fs, n_size=512)
        e_ffts.append(feature)
    e_fft = np.mean(e_ffts, axis=0)

    o_ffts = []
    for file in a_files:
        wave = thinkdsp.read_wave(filename="../data/01MDA/a.wav")
        fs, audio = read("../data/01MDA/a.wav")
        start, end = get_center_vowel(wave)
        segment_audio = get_segment_audio(audio, fs, start, end)
        feature = get_fft_of_segment_audio(segment_audio, fs, n_size=512)
        o_ffts.append(feature)
    o_fft = np.mean(o_ffts, axis=0)

    i_ffts = []
    for file in a_files:
        wave = thinkdsp.read_wave(filename="../data/01MDA/a.wav")
        fs, audio = read("../data/01MDA/a.wav")
        start, end = get_center_vowel(wave)
        segment_audio = get_segment_audio(audio, fs, start, end)
        feature = get_fft_of_segment_audio(segment_audio, fs, n_size=512)
        i_ffts.append(feature)
    i_fft = np.mean(i_ffts, axis=0)

    print(u_fft.shape)
    print(e_fft.shape)
    print(o_fft.shape)
    print(a_fft.shape)
    print(i_fft.shape)
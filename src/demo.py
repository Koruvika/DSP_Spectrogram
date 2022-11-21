import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
from scipy.io.wavfile import read
import numpy as np

from thinkdsp import Spectrogram, read_wave, decorate


def show_spectrogram_scipy(path="../data/01MDA/a.wav"):
    fs, audio = read(path)

    # spectrogram in scipy
    f, t, Sxx = signal.spectrogram(audio, fs)
    plt.pcolormesh(t, f, Sxx, shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def show_spectrogram_thinkdsp(path="../data/01MDA/a.wav"):
    # spectrogram in thinkdsp
    waves = read_wave(path)
    waves.normalize()
    gram = waves.make_spectrogram(seg_length=int(fs * 0.005))
    gram.plot(high=3000)
    decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')
    plt.show()


if __name__ == "__main__":
    fs, audio = read("../data/train/01MDA/a.wav")

    # spectrogram in scipy
    # show_spectrogram_scipy()

    # spectrogram in ThinkDSP
    show_spectrogram_thinkdsp()

    # ...

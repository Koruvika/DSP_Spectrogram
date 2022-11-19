import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
from scipy.io.wavfile import read
import numpy as np

if __name__ == "__main__":
    fs, audio = read("../data/01MDA/a.wav")
    spectrum = fft(audio)
    plt.plot(np.abs(spectrum))
    plt.show()

    # f, t, Sxx = signal.spectrogram(audio, fs)
    # plt.pcolormesh(t, f, Sxx, shading='auto')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    # fs, audio = read("../data/02FVA/o.wav")
    # spectrum = fft(audio)
    # plt.plot(spectrum)
    # plt.show()
    #
    # fs, audio = read("../data/03MAB/o.wav")
    # spectrum = fft(audio)
    # plt.plot(spectrum)
    # plt.show()

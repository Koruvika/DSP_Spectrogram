import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    plt.style.use("seaborn")
    # read training fft
    a_fft = np.load("a_fft.npy")
    u_fft = np.load("u_fft.npy")
    e_fft = np.load("e_fft.npy")
    i_fft = np.load("i_fft.npy")
    o_fft = np.load("o_fft.npy")

    fig, axs = plt.subplots(5)
    fig.suptitle('FFT Feature Extraction')
    axs[0].set_title(f"'u' vowel-FFT feature")
    axs[0].plot(u_fft)
    axs[1].set_title(f"'e' vowel-FFT feature")
    axs[1].plot(e_fft)
    axs[2].set_title(f"'o' vowel-FFT feature")
    axs[2].plot(o_fft)
    axs[3].set_title(f"'a' vowel-FFT feature")
    axs[3].plot(a_fft)
    axs[4].set_title(f"'i' vowel-FFT feature")
    axs[4].plot(i_fft)
    fig.tight_layout()
    plt.show()


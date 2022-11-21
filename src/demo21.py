import numpy as np

if __name__ == "__main__":
    # read training fft
    a_fft = np.load("a_fft.npy")
    u_fft = np.load("u_fft.npy")
    e_fft = np.load("e_fft.npy")
    i_fft = np.load("i_fft.npy")
    o_fft = np.load("o_fft.npy")

    # calculate test fft

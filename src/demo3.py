#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import librosa
from sklearn.cluster import KMeans
from SpeechSegment import silence_discrimination
from scipy.io.wavfile import read
from scipy import fft
import scipy.signal as sig
import thinkdsp
from sklearn.metrics import confusion_matrix
import os

K_CLUSTER=5

np.random.seed(0)

a_mfcc = np.load("a_mfcc.npy", allow_pickle=True)
u_mfcc = np.load("u_mfcc.npy", allow_pickle=True)
e_mfcc = np.load("e_mfcc.npy", allow_pickle=True)
i_mfcc = np.load("i_mfcc.npy", allow_pickle=True)
o_mfcc = np.load("o_mfcc.npy", allow_pickle=True)

#%%
def get_center_vowel(wave, file):
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


def get_mfcc(filename, N_MFCC):
    wave = thinkdsp.read_wave(filename=filename)
    fs, audio = read(filename)
    start, end = get_center_vowel(wave, filename)
    segment_audio = get_segment_audio(audio, fs, start, end)
    mfcc = librosa.feature.mfcc(y=segment_audio.astype('float64'), sr=fs, n_mfcc=N_MFCC, n_fft=512)
    feature=np.sum(mfcc, axis=1)
    feature_norm=(feature-np.mean(feature))/np.std(feature)
    return feature_norm


def k_mean(data, K_CLUSTER):
    kmeans = KMeans(init="k-means++", n_clusters=K_CLUSTER).fit(data)
    codebook = kmeans.cluster_centers_
    return codebook


a_mfcc_km = k_mean(a_mfcc, K_CLUSTER)
u_mfcc_km = k_mean(u_mfcc, K_CLUSTER)
i_mfcc_km = k_mean(i_mfcc, K_CLUSTER)
e_mfcc_km = k_mean(e_mfcc, K_CLUSTER)
o_mfcc_km = k_mean(o_mfcc, K_CLUSTER)


def save_mfcc_and_train_data(N_MFCC):
    vowel_files = []

    folders = os.listdir("../data/train")[1:-1]
    for folder in folders:
        files = os.listdir("../data/train/" + folder)
        for file in files:
            vowel_files.append("../data/train/" + folder + "/" + file)
            
    folders = os.listdir("../data/test")[1:-1]
    for folder in folders:
        files = os.listdir("../data/test/" + folder)
        for file in files:
            vowel_files.append("../data/test/" + folder + "/" + file)

    a_files = [file for file in vowel_files if file.split(".")[-2][-1] == "a"]
    u_files = [file for file in vowel_files if file.split(".")[-2][-1] == "u"]
    e_files = [file for file in vowel_files if file.split(".")[-2][-1] == "e"]
    o_files = [file for file in vowel_files if file.split(".")[-2][-1] == "o"]
    i_files = [file for file in vowel_files if file.split(".")[-2][-1] == "i"]

    a_mfcc = []
    for file in a_files:
        a_mfcc.append(get_mfcc(file, N_MFCC))
        
    # a_mfcc=np.hstack(a_mfcc).reshape(-1, 13, 20)
    a_mfcc=np.array(a_mfcc)

    u_mfcc = []
    for file in u_files:
        u_mfcc.append(get_mfcc(file, N_MFCC))
    u_mfcc=np.array(u_mfcc)
    
    e_mfcc = []
    for file in e_files:
        e_mfcc.append(get_mfcc(file, N_MFCC))
    e_mfcc=np.array(e_mfcc)

    o_mfcc = []
    for file in o_files:
        o_mfcc.append(get_mfcc(file, N_MFCC))
    o_mfcc=np.array(o_mfcc)

    i_mfcc = []
    for file in i_files:
        i_mfcc.append(get_mfcc(file, N_MFCC))
    i_mfcc=np.array(i_mfcc)

    print(u_mfcc.shape)
    print(e_mfcc.shape)
    print(o_mfcc.shape)
    print(a_mfcc.shape)
    print(i_mfcc.shape)

    with open("u_mfcc.npy", "wb") as f:
        np.save(f, u_mfcc)
    with open("e_mfcc.npy", "wb") as f:
        np.save(f, e_mfcc)
    with open("o_mfcc.npy", "wb") as f:
        np.save(f, o_mfcc)
    with open("a_mfcc.npy", "wb") as f:
        np.save(f, a_mfcc)
    with open("i_mfcc.npy", "wb") as f:
        np.save(f, i_mfcc)


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
        

def calculate_distance_mfcc(feature1, feature2):
    return np.sum(np.abs(feature1 - feature2)) / feature2.shape[0]

 
def inference(testfile, N_MFCC):
    feature = get_mfcc(testfile, N_MFCC)
    a_distance = calculate_distance_mfcc(feature, a_mfcc)
    u_distance = calculate_distance_mfcc(feature, u_mfcc)
    i_distance = calculate_distance_mfcc(feature, i_mfcc)
    e_distance = calculate_distance_mfcc(feature, e_mfcc)
    o_distance = calculate_distance_mfcc(feature, o_mfcc)
    
    # a_distance = calculate_distance_mfcc(feature, a_mfcc_km)
    # u_distance = calculate_distance_mfcc(feature, u_mfcc_km)
    # i_distance = calculate_distance_mfcc(feature, i_mfcc_km)
    # e_distance = calculate_distance_mfcc(feature, e_mfcc_km)
    # o_distance = calculate_distance_mfcc(feature, o_mfcc_km)

    # u: 0, e: 1, o: 2, a: 3, i: 4
    distance = np.array([u_distance, e_distance, o_distance, a_distance, i_distance])

    predict = np.argmin(distance)

    label = testfile.split(".")[-2][-1]

    return get_vowel_predict(predict), label
    
#%%
if __name__=="__main__":
    # save_mfcc_and_train_data()
    save_mfcc_and_train_data(13)
    vowel_files = []

    folders = os.listdir("../data/test")
    for folder in folders:
        files = os.listdir("../data/test/" + folder)
        for file in files:
            vowel_files.append("../data/test/" + folder + "/" + file)
    
    predict_results = []
    label_results = []
    for file in vowel_files:
        predict, label = inference(file,N_MFCC=13)
        predict_results.append(predict)
        label_results.append(label)
        
    conf = confusion_matrix(label_results, predict_results, labels=["u", "e", "o", "a", "i"])
    
    alphabets = ['u', 'e', 'o', 'a', 'i']
    figure = plt.figure()
    axes = figure.add_subplot(111)
    caxes = axes.matshow(conf, interpolation='nearest')
    figure.colorbar(caxes)
    for (i, j), z in np.ndenumerate(conf):
        axes.text(j, i, '{:d}'.format(z), ha='center', va='center')
    axes.set_xticklabels([''] + alphabets)
    axes.set_yticklabels([''] + alphabets)
    plt.show()
    print(f'Accuracy: {np.trace(conf)/np.sum(conf)}')
    
    
    
    
# %%
save_mfcc_and_train_data(13)
# %%

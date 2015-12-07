import scipy.io.wavfile as wavfile
import scipy
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import glob

root_folder = "/media/files/musicsamples/genres"
genre_list = ["classical", "jazz", "pop","rock","metal"]


def compute_fft_for_folder(folderpath):
    for filename in os.listdir(folderpath):
        real_path = os.path.join(folderpath,filename)
        if filename.endswith(".au"):
            create_fft(real_path)
        elif os.path.isdir(real_path):
            compute_fft_for_folder(real_path)


def create_fft(filename):
    X, sample_rate = librosa.load(filename)
    fft_features = abs(scipy.fft(X)[:1000])
    base_filename, extension = os.path.splitext(filename)
    data_filename = base_filename + '.fft'
    scipy.save(data_filename, fft_features)
    print filename, " fft'ed"

def read_fft(genre_list,
             base_dir='/media/files/musicsamples/genres'):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir,genre,"*.fft.npy")
        file_list = glob.glob(genre_dir)

        for fileName in file_list:
            fft_features = scipy.load(fileName)
            X.append(fft_features[:1000])
            y.append(label)
    return np.array(X), np.array(y)



compute_fft_for_folder(folderpath=root_folder)
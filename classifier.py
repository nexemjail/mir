import os
import numpy as np
import pandas as pd

from NN import NeuralNetwork
import time
from random import shuffle
'''
import matplotlib.pyplot as plt

import spark_initializer
import pyspark
'''
import scipy
import scipy.signal
import Sampler


def convert_audio_to_csv(path):
    for f in os.listdir(path):
        if f.endswith('.csv') or f.endswith(".npy"):
            continue
        if not f.endswith('.au'):
            place = path + '/' + f
            print "going to ", f
            convert_audio_to_csv(place)
        else:
            compute_treat_matrix(path, f)


def compute_treat_matrix(path, name):
    file_name = "/".join((path, name))
    duration, chunk_size = 29., .2

    unscaled_features = Sampler.convert(file_name, duration)
    frame = pd.DataFrame(unscaled_features)
    p = get_csv_path(path,name)
    frame.to_csv(p)
    print name, " converted and saved as ", path

    '''
    sampler = Sampler.Sampler(file_name, duration)

    plt.hold(False)
    splitted = np.array_split(sampler.mel_spectrum,duration/chunk_size, axis=1)
    for i in xrange(int(duration/chunk_size)):
        plt.plot(splitted[i])
        plt.show()

    #x,fs = librosa.load(file_name, duration=duration)
    '''
    '''
    spectogram = librosa.feature.spectral_bandwidth(x,sr= fs,)[0]
    plt.plot(spectogram)
    plt.show()
    pk = scipy.signal.find_peaks_cwt(spectogram, np.arange(0, 50))
    print spectogram[pk]
    tempo , beats = librosa.beat.beat_track(x, sr=fs)
    plt.plot(beats)
    plt.show()
    '''

    #mfcc_transform = librosa.feature.mfcc(x, sr=fs)
    #mfcc_transform = scale_features(mfcc_transform)
    #mean_values = mfcc_transform.mean(axis = 1)

    #plt.plot(mean_values)
    #plt.show()
    #frame = pd.DataFrame(mean_values)
    #p = get_csv_path(path, name, '_mean')
    #frame.to_csv(p)
    # TODO: take mean of mfcc in each frequency

    #a = librosa.feature.zero_crossing_rate(x)
    #s = librosa.feature.melspectrogram(x,sr = fs)
    #plt.hold(b = True)
    #plt.plot(mfcc_transform)
    #plt.show()


def scale_features(data):
    x_min = data.min(axis=0)
    x_max = data.max(axis=0)
    for index in xrange(data.shape[0]):
        data[index,:] = (data[index,:] - x_min)/(x_max - x_min)
    return data


def get_csv_path(path, filename,addition = "_treats", extra = ""):
    addition = "_treats29"
    extension = '.csv'
    filename = "".join(filename.split(".")[0:-1])
    return "{0}/{1}{2}{3}{4}".format(path,filename,addition,extra,extension)


def convert_all():
    path = "/media/files/musicsamples/genres"
    convert_audio_to_csv(path)


def read_fft_features(path, value):
    features = []
    t = time.time()
    for filename in os.listdir(path):
        if filename.endswith(".npy"):
            real_path = "{0}/{1}".format(path, filename)
            fft_features = scipy.load(real_path)
            features.append(np.asarray(fft_features[:1000]))
    print time.time() - t
    length = len(features)
    data = np.array(features)
    values = np.array([value] * length)
    return data, values


def read_features(path,value):
    features = []
    t = time.time()
    for filename in os.listdir(path):
        if filename.endswith("treats29.csv"):
            real_path = "{0}/{1}".format(path, filename)
            array = pd.DataFrame.from_csv(real_path).__array__()
            features.append(np.asarray(scale_features(array)).ravel())
    print time.time() - t
    length = len(features)
    data = np.array(features)
    values = np.array([value] * length)
    return data, values


if __name__ == "__main__":
    '''
    path = '/media/files/musicsamples/genres/pop'
    features, values = read_features(path,[1,0])
    path = '/media/files/musicsamples/genres/classical'
    '''

    path_rock = "/media/files/musicsamples/genres/rock"
    rock_data, rock_values = read_features(path_rock,[1,0])
    print rock_data.shape
    rock = zip(rock_data,rock_values)
    path_classic = "/media/files/musicsamples/genres/classical"
    class_data, class_value = read_features(path_classic,[0,1])
    classical = zip(class_data, class_value)
    extended_list = rock[:-10]
    extended_list.extend(classical[:-10])
    shuffle(extended_list)
    import sklearn.svm as svm
    nn = svm.SVC()

    nn = NeuralNetwork(learning_rate=0.00000000000000000000000000000001, num_of_hidden_layers=5, hidden_layer_size=75)
    data = [k[0] for k in extended_list]
    print type(data)
    values = [k[1] for k in extended_list]
    print type(values)
    nn.fit(data,values)
    for i in xrange(10):
        print nn.predict(rock[-i][0])
        print rock[-i][1], " pop"
        print nn.predict(classical[-i][0])
        print classical[-i][1], " classical"

    #nn.save_synapse_to_file("synapses")



    #convert_all()



    '''
    x, sr = librosa.load('/media/files/musicsamples/Cantaperme.wav')
    #fft_tranform = scipy.fft(x)
    #fft_tranform -= fft_tranform.mean()
    #plt.plot(fft_tranform)
    #plt.show()

    #onset_envelope = librosa.onset.onset_detect(x)
    #plt.plot(onset_envelope[0])
    #plt.show()

    tempo, beats = librosa.beat.beat_track(x,)
    plt.plot()
    plt.show()

    centroid = librosa.feature.spectral_centroid(x,)
    plt.plot(range(centroid.size), centroid.T)
    plt.show()
    librosa.feature.mfcc()
    roloff = librosa.feature.spectral_rolloff(x,)

    plt.plot(range(roloff.size),roloff.T)
    plt.show()

    #spectrogram = scipy.fft(x)
    #spectral_flux = librosa.feature.spectral_contrast(spectrogram,n_bands=1)
    #plt.plot(spectral_flux)
    #plt.show()
    #k = 8
    zero_crossing_rate = librosa.feature.zero_crossing_rate(x)
    zcr_for_file = zero_crossing_rate.sum()
    print zcr_for_file
    '''
    '''
    path_rock = "/media/files/musicsamples/genres/rock"
    rock_data, rock_values = read_fft_features(path_rock,[1,0])
    rock = zip(rock_data,rock_values)
    path_classic = "/media/files/musicsamples/genres/classical"
    class_data, class_value = read_fft_features(path_classic,[0,1])
    classical = zip(class_data, class_value)
    extended_list = rock[:-10]
    extended_list.extend(classical[:-10])
    shuffle(extended_list)
    nn = NeuralNetwork(learning_rate=0.01, num_of_hidden_layers=5, hidden_layer_size=75)
    data = [k[0] for k in extended_list]
    values = [k[1] for k in extended_list]
    nn.fit(data,values)
    for i in xrange(10):
        print nn.predict(rock[-i][0])
        print rock[-i][1], " pop"
        print nn.predict(classical[-i][0])
        print classical[-i][1], " classical"

    #nn.save_synapse_to_file("synapses")

    path = "/media/files/musicsamples/genres"
    current_path = path+"/jazz/jazz00000.csv"
    data_frame = load_from_csv(current_path)
    arr = data_frame.__array__()
    for line_number in xrange(arr.shape[0]):
        print line_number,": ",min(arr[line_number]), max(arr[line_number]), np.mean(arr[line_number])
    '''






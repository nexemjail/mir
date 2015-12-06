import os
import numpy as np
import pandas as pd
import time
from random import shuffle
import scipy
import scipy.signal
import Sampler
from spark_initializer import init_spark, get_spark_context
import data_loader as loader
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as conf_matrix
import pylab as pl

def convert_audio_to_csv(path):
    for f in os.listdir(path):
        if f.endswith('.csv') or f.endswith(".npy"):
            continue
        if not f.endswith('.au'):
            place = path + '/' + f
            print "going to ", f
            convert_audio_to_csv(place)
        else:
            compute_feature_matrix(path, f)


def compute_feature_matrix(path, name):
    file_name = "/".join((path, name))
    duration, half_part_length = 30., 0.03
    print "started to converting " + file_name
    t = time.time()
    unscaled_features = Sampler.convert(file_name, duration,half_part_length)
    frame = pd.DataFrame(unscaled_features)
    p = get_csv_path(path,name,'_features30new')
    frame.to_csv(p)
    print name, " converted in " ,time.time() - t," saved as", p

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

    #a = librosa.feature.zero_crossing_rate(x)
    #s = librosa.feature.melspectrogram(x,sr = fs)
    #plt.hold(b = True)
    #plt.plot(mfcc_transform)
    #plt.show()


def get_csv_path(path, filename,addition = "_treats", extra = ""):
    extension = '.csv'
    filename = "".join(filename.split(".")[0:-1])
    return "{0}/{1}{2}{3}{4}".format(path,filename,addition,extra,extension)


def get_all_file_paths(folderName):
    path_list = []
    for f in os.listdir(folderName):
        if f.endswith(".au"):
            path_list.append([folderName, f])
    return path_list


def convert_all(list_of_dirs, use_spark = False):
    if use_spark:
        init_spark()
        sc = get_spark_context()
    path = "/media/files/musicsamples/genres/"
    for el in list_of_dirs:
        path_list = get_all_file_paths(path + '/' + el)
        if not use_spark:
            [compute_feature_matrix(y[0],y[1]) for y in path_list]
        else:
            rdd = sc.parallelize(path_list).cache()
            rdd.map(lambda y: compute_feature_matrix(y[0],y[1])).collect()


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


def create_sets(batch_size, array_of_dv_sets):
    training_set = []
    test_set = []
    for arg in array_of_dv_sets:
        training_set.extend(arg[:batch_size])
        test_set.extend(arg[:-batch_size])
    shuffle(training_set)
    return training_set, test_set


def map_dataset(dataset, mapper):
    mapped_dataset = []
    for element in dataset:
        mapped_dataset.append((element[0], mapper[element[1]]))
    return mapped_dataset


def separate(dataset):
    return [t[0] for t in dataset], [t[1] for t in dataset]


def test_it():

    dataset_size = 100
    test_size = 10

    dataloader = loader.Loader(['rock','classical'],'30new.csv','/media/files/musicsamples/genres')
    datasets = dataloader.get_dataset()
    dv_pairs = []
    for v in datasets.values():
        dv_pairs.append(loader.combine_to_data_value_pair(v[0], v[1]))

    genre_mapper = OrderedDict([
        ('rock',0),
        ('classical',1),
        ('blues',2),
        ('pop',3),
        ('metal',4),
        ('reggae',5),
        ('country',6),
        ('disco',7),
        ('jazz',8)
    ])

    training_set, test_set = create_sets(dataset_size - test_size, dv_pairs)
    training_set = map_dataset(training_set, genre_mapper)
    test_set = map_dataset(test_set, genre_mapper)
    import sklearn.svm as svm

    nn = svm.SVC()

    training_data, training_values = separate(training_set)
    nn.fit(training_data, training_values)

    N = len(genre_mapper)
    confusion_matrix = np.zeros((N, N))

    cm = conf_matrix([nn.predict(item[0])for item in test_set], [item[1] for item in test_set])
    pl.matshow(cm)
    pl.title('Confusion matrix of the classifier')
    pl.colorbar()
    pl.show()
    accuracy = [None] * N
    for i in xrange(N):
        true = confusion_matrix[i,i]
        num_of_results = np.sum(confusion_matrix[i,:])
        if num_of_results != 0:
            accuracy[i] = float(true)/num_of_results * 100

    for item in test_set:
        prediction = nn.predict(item[0])
        real_value = item[1]
        confusion_matrix[real_value, prediction] += 1

    print genre_mapper.keys()
    print genre_mapper.values()
    genre_list = [k for (k,v) in genre_mapper.iteritems()]
    for index in xrange(N):
        print confusion_matrix[index,:],
        print genre_list[index]
    plt.imshow(confusion_matrix, interpolation='nearest')
    plt.xticks(range(N), genre_list)
    plt.yticks(range(N), genre_list)
    plt.show()



if __name__ == "__main__":
    #convert_all(['hiphop','jazz','rock','blues','metal','pop','classical','reggae','disco','country'], False)
    test_it()

    #nn.save_synapse_to_file("synapses")
    #spudi lalka sasai
    #Will See how to sasi





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






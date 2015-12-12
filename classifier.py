import os
import numpy as np
import pandas as pd
import time
from random import shuffle
import scipy
import scipy.signal
from Sampler import convert
from spark_initializer import init_spark, get_spark_context
from data_loader import Loader
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as conf_matrix
import pylab as pl
import sklearn.svm as svm
import helper
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.externals import joblib

import audiotools


def get_genre_mapper():
    return OrderedDict([
            ('rock', 0),
            ('classical', 1),
            ('blues', 2),
            ('pop', 3),
            ('metal', 4),
            ('country', 5),
            ('disco', 6),
            ('jazz', 7)
        ])

def get_genre_unmapper():
    mapper = get_genre_mapper()
    genre_unmapper = OrderedDict()
    for (k,v) in mapper.iteritems():
        genre_unmapper[v] = k
    return genre_unmapper


def predict_genre(classifier, prediction_vector, genre_unmapper):
    prediction = classifier.predict(prediction_vector)[0]
    return genre_unmapper[prediction]


def convert_au_to_csv(path):
    for f in os.listdir(path):
        if f.endswith('.csv') or f.endswith(".npy"):
            continue
        if not f.endswith('.au'):
            place = path + '/' + f
            print "going to ", f
            convert_au_to_csv(place)
        else:
            compute_feature_matrix(path, f)


def compute_feature_matrix(path, name):
    file_name = "/".join((path, name))
    duration, half_part_length = 30., 0.03
    print "started to converting " + file_name
    t = time.time()
    unscaled_features = convert(file_name, duration,half_part_length,offset=0, num_processes=1)
    frame = pd.DataFrame(unscaled_features)
    p = construct_csv_path(path, name, '_features30new')
    frame.to_csv(p)
    print name, " converted in " ,time.time() - t," saved as", p


def construct_csv_path(path, filename, addition ="", extra =""):
    extension = '.csv'
    filename = "".join(filename.split(".")[0:-1])
    return "{0}/{1}{2}{3}{4}".format(path,filename,addition,extra,extension)


def get_au_files(folderName):
    path_list = []
    for f in os.listdir(folderName):
        if f.endswith(".au"):
            path_list.append([folderName, f])
    return path_list


def convert_all_au_in_directory(list_of_dirs, use_spark = False):
    if use_spark:
        init_spark()
        sc = get_spark_context()
    path = "/media/files/musicsamples/genres/"
    for el in list_of_dirs:
        path_list = get_au_files(path + '/' + el)
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


def fit_classifier(classifier, training_data, training_values):
    classifier.fit(training_data, training_values)


def plot_confusion_matrix(classifier, test_set):
    cm = conf_matrix([classifier.predict(item[0]) for item in test_set], [item[1] for item in test_set])
    pl.matshow(cm)
    pl.title('Confusion matrix of ' + str(type(classifier)))
    pl.colorbar()
    pl.show()


def load_classifiers(root_dir = 'saved_classifiers'):
    paths = []
    for i in xrange(1,4):
        paths.append(
            os.path.join(root_dir, "classifier{0}.joblib".format(str(i)))
            )
    classifiers = []
    for path in paths:
        classifiers.append(joblib.load(path))
    return classifiers


def save_classifiers(classifiers, root_dir = 'saved_classifiers'):
    for (i,c) in enumerate(classifiers):
        joblib.dump(c, os.path.join(root_dir,"classifier{0}.joblib".format(str(i+1))))


def get_prediction_vector(path, duration = 20, offset = 30):
        features = convert(path,duration=duration, offset=offset)
        scaled_features = helper.scale_features(features)
        variance = helper.deviation(scaled_features)
        mean = helper.average(scaled_features)
        prediction_vector = helper.combine_mean_and_variance(mean,variance)
        return prediction_vector

def test_it(path = None):
    genre_mapper = get_genre_mapper()
    genre_unmapper = get_genre_unmapper()

    if path is None:
        dataset_size = 100
        test_size = 20

        dataloader = Loader(['rock','classical','jazz', 'blues','disco','country','pop','metal'],
                            '30new.csv','/media/files/musicsamples/genres')
        datasets = dataloader.get_dataset()
        dv_pairs = []

        for v in datasets.values():
            dv_pairs.append(helper.combine_to_data_value_pair(v[0], v[1]))

        training_set, test_set = create_sets(dataset_size - test_size, dv_pairs)
        training_set = map_dataset(training_set, genre_mapper)
        test_set = map_dataset(test_set, genre_mapper)

        N = len(genre_mapper)
        support_vector_machine = svm.SVC(C = 10**5)
        kmeans = KMeans(n_clusters=N)
        bayes = GaussianNB()

        classifiers = [support_vector_machine,kmeans,bayes]

        training_data, training_values = helper.separate_input_and_check_values(training_set)

        #classifiers = load_classifiers()
        for c in classifiers:
            fit_classifier(c,training_data, training_values)
            plot_confusion_matrix(c,test_set)

        save_classifiers(classifiers)
    else:
        t = time.time()
        prediction_vector = get_prediction_vector(path)
        print time.time() - t
        classifiers = load_classifiers()
        predictions= []
        for c in classifiers:
            predictions.append(
                predict_genre(c,prediction_vector,genre_unmapper))

        print predictions
        print [type(c) for c in classifiers]




    '''
    confusion_matrix = np.zeros((N, N))

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

    print test_set[0][0], type(test_set[0][0]), test_set[0][0].shape
    '''

    #genre_list = [k for (k,v) in genre_mapper.iteritems()]
    #for index in xrange(N):
    #    print confusion_matrix[index,:],
    #    print genre_list[index]
    #plt.imshow(confusion_matrix, interpolation='nearest')
    #plt.xticks(range(N), genre_list)
    #plt.yticks(range(N), genre_list)
    #plt.show()



if __name__ == "__main__":
    '''
    convert_all_au_in_directory(
        ['hiphop','jazz','rock','blues','metal','pop','classical','reggae','disco','country'],
        use_spark=False)
    test_it()
    '''
    testfile = '/media/files/musicsamples/converted_to_au/Rick Astley - Never gonna give you up (2001 ver).au'
    test_it(testfile)
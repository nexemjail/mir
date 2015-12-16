import os
import time
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as conf_matrix
import pylab as pl
import sklearn.svm as svm
import helper
import numpy as np
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from data_loader import Loader
from helper import get_genre_unmapper
from process_parallelizer import convert_with_processes


def predict_genre(classifier, prediction_vector, genre_unmapper):
    prediction = classifier.predict(prediction_vector)[0]
    return genre_unmapper[prediction]


def create_sets(batch_size, array_of_dv_sets):
    training_set = []
    test_set = []
    for arg in array_of_dv_sets:
        training_set.extend(arg[:batch_size])
        test_set.extend(arg[:-batch_size])
    shuffle(training_set)
    return training_set, test_set


def map_to_numeric_dataset(dataset, mapper):
    mapped_dataset = []
    for element in dataset:
        mapped_dataset.append((element[0], mapper[element[1]]))
    return mapped_dataset


def fit_classifier(classifier, training_data, training_values):
    classifier.fit(training_data, training_values)


def plot_confusion_matrix(classifier, test_set):
    cm = conf_matrix([classifier.predict(item[0]) for item in test_set], [item[1] for item in test_set])
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    pl.matshow(cm)
    marks = range(8)
    genre_unmap = get_genre_unmapper()

    plt.xticks(marks, [genre_unmap[m] for m in marks], rotation = 90)
    plt.yticks(marks, [genre_unmap[m] for m in marks])

    pl.colorbar()
    pl.show()


def load_svm(root_dir = 'saved_classifiers'):
    path =  os.path.join(root_dir, "classifier{0}.joblib".format(1))
    return joblib.load(path)


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


def get_prediction_vector(path, duration = 20, offset = 30, half_part_length = 0.05):
        features = convert_with_processes(path, duration=duration, offset=offset,half_part_length=half_part_length)
        scaled_features = helper.scale_features(features)
        variance = helper.deviation(scaled_features)
        mean = helper.average(scaled_features)
        prediction_vector = helper.combine_mean_and_variance(mean,variance)
        return prediction_vector


def train_classifiers():

    genre_mapper = helper.get_genre_mapper()

    dataset_size = 100
    test_size = 20
    dataloader = Loader(['rock','classical','jazz', 'blues','disco','country','pop','metal'],
                       '30new.csv','/media/files/musicsamples/genres')

    #dataloader = Loader(['rock','classical','jazz', 'blues','disco','country','pop','metal'],
    #                    '_mfcc_scaled.csv','/media/files/musicsamples/genres')
    datasets = dataloader.get_dataset()
    dv_pairs = []

    for v in datasets.values():
        dv_pairs.append(helper.combine_to_data_value_pair(v[0], v[1]))

    training_set, test_set = create_sets(dataset_size - test_size, dv_pairs)
    training_set = map_to_numeric_dataset(training_set, genre_mapper)
    test_set = map_to_numeric_dataset(test_set, genre_mapper)

    N = len(genre_mapper)
    support_vector_machine = svm.SVC(C = 10**5)
    kmeans = KMeans(n_clusters=N)
    bayes = GaussianNB()

    classifiers = [support_vector_machine,kmeans,bayes]

    training_data, training_values = helper.separate_input_and_check_values(training_set)

    for c in classifiers:
        fit_classifier(c,training_data, training_values)
        plot_confusion_matrix(c,test_set)

    save_classifiers(classifiers)


def predict(path):
    t = time.time()
    prediction_vector = get_prediction_vector(path)
    print time.time() - t
    classifiers = load_classifiers()
    predictions= []
    for c in classifiers:
        predictions.append(
            predict_genre(c,prediction_vector,genre_unmapper=get_genre_unmapper()))

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


if __name__ == "__main__":
    train_classifiers()
    testfile = '/media/files/musicsamples/converted_to_au/Rick Astley - Never gonna give you up (2001 ver).au'
    predict(testfile)
import numpy as np
from collections import OrderedDict


def get_genre_mapper():
    """
    :rtype: OrderedDict
    """
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


def separate_input_and_check_values(dataset):
    return [t[0] for t in dataset], [t[1] for t in dataset]


def combine_to_data_value_pair(data, values):
    return zip(data, values)


def average(data):
    return data.mean(axis = 0)


def deviation(data):
    return data.var(axis = 0)


def combine_mean_and_variance(mean, variance):
    result_array = np.zeros(shape=(len(mean) + len(variance)))
    for index in xrange(len(result_array)):
        if index % 2 == 0:
            result_array[index] = mean[index // 2]
        else:
            result_array[index] = variance[index // 2]
    return result_array


def scale_features(data):
    x_min = data.min(axis=0)
    x_max = data.max(axis=0)
    for index in xrange(data.shape[0]):
        data[index,:] = (data[index,:] - x_min)/(x_max - x_min)
    return data
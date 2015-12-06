import numpy as np
import pandas as pd
import time
import os


def combine_to_data_value_pair(data, values):
    return zip(data, values)


def average(data):
    return data.mean(axis = 0)


def deviation(data):
    return data.var(axis = 0)


def combine(x,y):
    result_array = np.zeros(shape=(len(x) + len(y)))
    for index in xrange(len(result_array)):
        if index % 2 == 0:
            result_array[index] = x[index//2]
        else:
            result_array[index] = y[index//2]
    return result_array


def scale_features(data):
    x_min = data.min(axis=0)
    x_max = data.max(axis=0)
    for index in xrange(data.shape[0]):
        data[index,:] = (data[index,:] - x_min)/(x_max - x_min)
    return data

def read_features_from_path_list(path_list):
    features = []
    t = time.time()
    for path in path_list:
        array = pd.DataFrame.from_csv(path).__array__()
        array = scale_features(array)
        sd = np.asarray(deviation(array)).ravel()
        means = np.asarray(average(array)).ravel()
        vector = combine(means, sd)
        features.append(vector)
    print "reading took ", time.time() - t
    return np.array(features)


def _get_names_list(directory, name_template):
    path_list = []
    for f in os.listdir(directory):
        if f.endswith(name_template):
            path_list.append(directory + '/' + f)
    return path_list



class Loader(object):
    def __init__(self, dir_list, name_template, source_path):
        self.features = dict()
        self.files_by_directory = dict()
        for directory_name in dir_list:
            full_dir_path = os.path.join(source_path, directory_name)
            self.files_by_directory[directory_name] = _get_names_list(full_dir_path, name_template)

    def get_dataset(self):
        """
        :return: dict with keys
        """
        return self._read_features()

    def _read_features(self):
        data_value_pair_dict = dict()
        for (folder_name, path_list) in self.files_by_directory.iteritems():
            features = read_features_from_path_list(path_list)
            values = np.array([folder_name] * len(features))
            data_value_pair_dict[folder_name] = [features, values]
        return data_value_pair_dict



if __name__ == "__main__":
    path = '/media/files/musicsamples/genres/'
    loader = Loader(['rock','pop'],'30new.csv',path)
    datasets = loader.get_dataset()
    print datasets['rock']
    print datasets['pop']




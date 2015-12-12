import numpy as np
import pandas as pd
import time
import os
import helper
import scipy


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


def read_features_from_path_list(path_list):
    features = []
    t = time.time()
    for path in path_list:
        array = pd.DataFrame.from_csv(path).__array__()
        array = helper.scale_features(array)
        variance = np.array(helper.deviation(array)).ravel()
        means = np.array(helper.average(array)).ravel()
        vector = helper.combine_mean_and_variance(means, variance)
        features.append(vector)
    print "reading took ", time.time() - t
    return np.array(features)


def get_names_list(directory, name_template):
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
            self.files_by_directory[directory_name] = get_names_list(full_dir_path, name_template)

    def get_dataset(self):
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



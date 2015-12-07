import os
import numpy as np
import pandas as pd
from librosa import feature, load
from NN import NeuralNetwork
import time
import DataValuePair


def convert_audio_to_csv(path):
    for f in os.listdir(path):
        if f.endswith('.csv'):
            continue
        if not f.endswith('.au'):
            place = path + '/' + f
            print "going to ", f
            convert_audio_to_csv(place)
        else:
            make_mffc(path,f)


def make_mffc(path, name):
    file_name = "/".join((path,name))
    x,fs = load(file_name)
    mfcc_transform = feature.mfcc(x, sr=fs)
    mfcc_transform = scale_features(mfcc_transform)
    frame = pd.DataFrame(mfcc_transform)
    p = get_csv_path(path,name)
    frame.to_csv(p)
    print name, " converted"


def scale_features(data):
    x_min = data.min(axis=1)
    x_max = data.max(axis=1)
    for index in xrange(data.shape[1]):
        data[:, index] = (data[:, index] - x_min)/(x_max - x_min)
    return data


def dataframe_from_csv(path):
    data_frame = pd.DataFrame.from_csv(path)
    return data_frame


def get_csv_path(path, filename):
    addition = "_mfcc"
    extension = '.csv'
    filename = "".join(filename.split(".")[0:-1])
    return "{0}/{1}{2}{3}".format(path,filename,addition,extension)


def convert_all():
    path = "/media/files/musicsamples/genres"
    convert_audio_to_csv(path)

def read_features(path):
    features = []
    t = time.time()
    i = 0
    for filename in os.listdir(path):
        if filename.endswith("scaled.csv"):
            real_path = "{0}/{1}".format(path, filename)
            array = dataframe_from_csv(real_path).__array__()
            features.append(np.asarray(array[:,:5]).ravel())
    print time.time() - t
    length = len(features)
    data = np.array(features)
    values = np.array([[1,0]] * length)
    return DataValuePair.DataValuePair(data,values)





if __name__ == "__main__":
    #convert_all()
    #audio = IPython.display.Audio('/media/files/musicsamples/Cantaperme.wav')
    path = "/media/files/musicsamples/genres/rock"
    fv = read_features(path)
    nn = NeuralNetwork(learning_rate=50, num_of_hidden_layers=2)
    nn.fit(fv.data[:-10],fv.values[:-10])
    for i in xrange(10):
        print nn.predict(fv.data[i])
        print fv.values[i]
    #nn.save_synapse_to_file("synapses")

    '''
    path = "/media/files/musicsamples/genres"
    current_path = path+"/jazz/jazz00000.csv"
    data_frame = load_from_csv(current_path)
    arr = data_frame.__array__()
    for line_number in xrange(arr.shape[0]):
        print line_number,": ",min(arr[line_number]), max(arr[line_number]), np.mean(arr[line_number])
    '''




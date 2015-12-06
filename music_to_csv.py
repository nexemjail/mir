from spark_initializer import init_spark
from spark_initializer import get_spark_context
import pandas as pd
import os
from Sampler import Sampler
import time
# GOVNOCLASS!!!!!!

class CSVConverter(object):
    def __init__(self, dirs, use_spark, duration = 30.0, half_fragment_duration = 0.06):
        self.convert_all(dirs, use_spark)
        self.duration = duration
        self.half_fragment_duration = half_fragment_duration

    def get_csv_path(self,path, filename,addition = "_treats", extra = ""):
        extension = '.csv'
        filename = "".join(filename.split(".")[0:-1])
        return "{0}/{1}{2}{3}{4}".format(path,filename,addition,extra,extension)


    def get_all_file_paths(self,folderName):
        path_list = []
        for f in os.listdir(folderName):
            if f.endswith(".au"):
                path_list.append([folderName, f])
        return path_list


    def convert_all(self,list_of_dirs, use_spark = False):
        if use_spark:
            init_spark()
            sc = get_spark_context()
        path = "/media/files/musicsamples/genres/"
        for el in list_of_dirs:
            path_list = self.get_all_file_paths(path + '/' + el)
            if not use_spark:
                [self.compute_feature_matrix(y[0],y[1]) for y in path_list]
            else:
                rdd = sc.parallelize(path_list).cache()
                rdd.map(lambda y: self.compute_feature_matrix(y[0],y[1])).collect()

    def convert_audio_to_csv(self,path):
        for f in os.listdir(path):
            if f.endswith('.csv') or f.endswith(".npy"):
                continue
            if not f.endswith('.au'):
                place = path + '/' + f
                print "going to ", f
                self.convert_audio_to_csv(place)
            else:
                self.compute_feature_matrix(path, f)


    def compute_feature_matrix(self,path, name):
        file_name = "/".join((path, name))
        duration, half_part_length = 30., 0.03
        print "started to converting " + file_name
        t = time.time()
        unscaled_features = Sampler.convert(file_name, duration,half_part_length)
        frame = pd.DataFrame(unscaled_features)
        p = self.get_csv_path(path,name,'_features30new')
        frame.to_csv(p)
        print name, " converted in " ,time.time() - t," saved as", p
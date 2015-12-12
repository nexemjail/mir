import os
import time
from process_parallelizer import convert_with_processes
import pandas as pd
from spark_initializer import init_spark, get_spark_context


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


def compute_feature_matrix(path, name, use_spark=False):
    file_name = "/".join((path, name))
    duration, half_part_length = 30., 0.03

    print "started to converting " + file_name
    t = time.time()
    if use_spark:
        unscaled_features = convert_with_processes(file_name, duration, half_part_length, offset=0, num_processes=1)
    else:
        unscaled_features = convert_with_processes(file_name, duration, half_part_length, offset=0)
    p = construct_csv_path(path, name, '_features30new')
    frame = pd.DataFrame(unscaled_features)
    frame.to_csv(p)

    print name, " converted in " ,time.time() - t," saved as", p


def construct_csv_path(path, filename, addition ="", extra =""):
    extension = '.csv'
    filename = "".join(filename.split(".")[0:-1])
    return "{0}/{1}{2}{3}{4}".format(path,filename,addition,extra,extension)


def get_all_audio_files_in_directory(path):
    path_list = []
    for f in os.listdir(path):
        if f.endswith(".au"):
            path_list.append([path, f])
    return path_list


def convert_all_audio_in_directory(list_of_dirs, use_spark = False):
    if use_spark:
        init_spark()
        sc = get_spark_context()

    path = "/media/files/musicsamples/genres/"
    for el in list_of_dirs:
        path_list = get_all_audio_files_in_directory(path + '/' + el)
        if not use_spark:
            [compute_feature_matrix(y[0], y[1]) for y in path_list]
        else:
            rdd = sc.parallelize(path_list).cache()
            rdd.map(lambda x: compute_feature_matrix(x[0],x[1], use_spark=use_spark)).collect()


if __name__ == "__main__":
     # TODO : do not uncomment this or be broken!
     '''

     convert_all_audio_in_directory(
        ['jazz','rock','blues','metal','pop','classical','disco','country'],
        use_spark=True)
     '''
 # -*- coding: utf8 -*-
from pyspark import SparkContext, SparkConf
import spark_pathfinder


def init_spark():
    return spark_pathfinder.init_spark()


def get_spark_context():
    conf = SparkConf().set("spark.default.parallelism", '128')\
    .set("spark.executor.instances", '8')\
    .set("spark.executor.memory", "8g")\
    .set("spark.python.worker.memory",'512mb')\
    .setMaster('master')
    sc = SparkContext('local[*]',conf = conf,)
    return sc


'''''
    раскрыть методы(чтоб побольше было)
    запараллелиить сэмплинг по частям на тредах(чтобы были треды)
'''''

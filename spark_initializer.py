import os
import sys
from pyspark import SparkContext, SparkConf
import spark_path

def init_spark():
	return spark_path.init_spark()


def get_spark_context():
    conf = SparkConf().set("spark.default.parallelism",4)\
    .set("spark.executor.instances", 4)\
    .set("spark.executor.memory", "2g")\
    .setMaster('master')
    sc = SparkContext('local[*]',conf = conf,)
    return sc
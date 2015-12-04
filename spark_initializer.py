import os
import sys
from pyspark import SparkContext, SparkConf

def init_spark():
    os.environ['SPARK_HOME'] = "/home/nexemjail/spark-1.5.2-bin-hadoop2.6"
    sys.path.append("/home/nexemjail/spark-1.5.2-bin-hadoop2.6/python")


def get_spark_context():
    conf = SparkConf().set("spark.default.parallelism",4)\
    .set("spark.executor.instances", 4)\
    .set("spark.executor.memory", "2g")\
    .setMaster('master')
    sc = SparkContext('local[*]',conf = conf,)
    return sc
import os
import sys


def init_spark():
    os.environ['SPARK_HOME'] = "/home/nexemjail/spark-1.5.2-bin-hadoop2.6"
    sys.path.append("/home/nexemjail/spark-1.5.2-bin-hadoop2.6/python")

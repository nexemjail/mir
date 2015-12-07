from pyspark import SparkContext, SparkConf
from spark_initializer import init_spark
import time
# add path to spark
textFile = "text.txt"
#testpath = "/media/files/Programming/Py/test/new.txt"
#sourceTextFile = "testFile.txt"
#generator.generate_numeric_file(sourceTextFile, 10 ** 6)
init_spark()
conf = SparkConf().set("spark.default.parallelism",4)\
    .set("spark.executor.instances", 4)\
    .set("spark.executor.memory", "2g")\
    .setMaster("master")


'''
size = 10 ** 7
#generator.generate_text_file(textFile,size)
sc = SparkContext('local', conf= conf)
t = time.time()
text = sc.textFile(textFile).cache()
counts = text.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1))
print counts.count(), time.time() - t
sc.stop()


t = time.time()
with open(textFile,'r') as f:
    text = f.read(size)
    splitted = text.split(' ')
    collection = map(lambda word: (word, 1),splitted)
print len(collection), time.time() - t
'''


'''
nums = sc.textFile(sourceTextFile,8).map(lambda x: int(x))
for i in range(5):
    nums = nums.map(lambda num: 1.0 / (1 + exp(-1*num)))

with open('new.txt', 'w')as f:
    for el in nums.collect():
        f.write(str(el) + '\n')
print str(True), time.time() - t


t = time.time()
collection = []
with open(sourceTextFile,'r') as f:
    for line in f:
        collection.append(int(line))

for i in range(5):
    for j in xrange(len(collection)):
        collection[j] = 1.0 / (1 + exp(-collection[j]))
with open("new1.txt",'w') as dist:
    for el in collection:
        dist.write(str(el)+'\n')
print str(True),time.time() - t
'''









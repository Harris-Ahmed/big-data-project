import sys
import json
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

conf = SparkConf()
conf.setAppName("Big Data")
sc = SparkContext(conf=conf)
stream = StreamingContext(sc, 2)

datastream = stream.socketTextStream("localhost", 6100)

x = datastream.flatMap(lambda datastream: json.loads(datastream))

x.print()

stream.start()
stream.awaitTermination()

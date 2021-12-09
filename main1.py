
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col


ab=SparkContext.getOrCreate()
ab.setLogLevel('OFF')
abb = StreamingContext(ab,1)
ssparks=SparkSession(ab)

try:
        record = abb.socketTextStream('localhost',6100)
except Exception as e:
        print(e)
        
def readstream(rdd):
        if(len(rdd.collect())>0):
          df=ssparks.read.json(rdd)
          df.show()
               
record.foreachRDD(lambda x:readstream(x))

abb.start()
abb.awaitTermination()

import json
import time
import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
import numpy as nmpy
import pandas as pnd
from textproc import *
import csv


from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.cluster import KMeans

if sys.argv[1] == 'train':
	headder = ["SVM_class","NB_Class","RF_class"]
	with open("train_track.csv",'w',newline='') as f:
		writer = csv.writer(f)
		writer.writerow(headder)

if sys.argv[1]=='cluster':
	clus_headder=['centroid1_diff', 'centroid2_diff']
	with open("cluster_track.csv",'w',newline='') as f1:
		writer = csv.writer(f1)
		writer.writerow(clus_headder)

owld1=nmpy.array([0,0])
owld2=nmpy.array([0,0])

sc1 = SparkContext.getOrCreate()
sc1.setLogLevel('OFF')
ssc1 = StreamingContext(sc1, 1)
sparks = SparkSession(sc1)

def outr(rdd):
	#print(rdd)
	return json.loads(rdd)

def func(rdd):
	#print(rdd.collect())
	daata=rdd.collect()
	if len(daata)==0:
		pass
	else:
		dframe= pnd.DataFrame(daata[0]).transpose()
		dframe=preprocess(dframe)
		#print(dframe)
		#teafidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords_english)  
		#x1 = teafidfconverter.fit_transform(dframe['feature1']).toarray()
		#y1 = dframe['feature0'].apply(lambda x1:int(x1))
		
		if sys.argv[1]=='train':
			teafidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords_english)  
			x1 = teafidfconverter.fit_transform(dframe['feature1']).toarray()
			y1 = dframe['feature0'].apply(lambda x1:int(x1))
			X_train, X_test, y_train, y_test = train_test_split(x1,y1,test_size = 0.05, random_state =26105111)
			sc11=svm_classifier(X_train, X_test, y_train, y_test)
			sc222=NB_classifier(X_train, X_test, y_train, y_test)
			sc333=Rf_classifier(X_train, X_test, y_train, y_test)
			print("Score c1: %s  ,Score c2: %s  ,Score: %s  "%(str(sc11),str(sc222),str(sc333)))
			with open("train_track.csv",'a',newline='') as f:
				writer = csv.writer(f)
				daata=[sc11,sc222,sc333]
				writer.writerow(daata)
		
		elif sys.argv[1]=='cluster':
			global owld1
			global owld2
			teafidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords_english)  
			x1 = teafidfconverter.fit_transform(dframe['feature1']).toarray()
			y1 = dframe['feature0'].apply(lambda x1:int(x1))
			X_train, X_test, y_train, y_test = train_test_split(x1,y1,test_size = 0.05, random_state =26105111)
			kmea = KMeans(n_clusters=2, init='random',n_init=10, max_iter=5, tol=1e-04, random_state=0)
			y_kmea = kmea.fit_predict(X_train)
			new111= kmea.cluster_centers_[:,0]
			new222= kmea.cluster_centers_[:,1]
			dist111= nmpy.linalg.norm(new111 - owld1)
			dist222= nmpy.linalg.norm(new222 - owld2)
			daata = [dist111,dist222]
			print(daata)
			with open("cluster_track.csv",'a',newline='') as f1:
				writer = csv.writer(f1)
				writer.writerow(daata)
			
			if min(daata) <= 0.0005:
				print("###########  small value of centroid shift detected ########################")
				
			owld1=new111
			owld2=new222
			
			
			
		else:
			teafidfconverter = TfidfVectorizer(max_features=275, min_df=5, max_df=0.7, stop_words=stopwords_english) 
			X_test= teafidfconverter.fit_transform(dframe['feature1']).toarray()
			y_test= dframe['feature0'].apply(lambda x1:int(x1))
			test_cycle(X_test,y_test)
			
		
	
daata = ssc1.socketTextStream("localhost", 6100)


j_data= daata.map(outr).foreachRDD(func)


ssc1.start()
ssc1.awaitTermination()

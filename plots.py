import sys
import numpy as nmpy
import pandas as pnd
import matplotlib.pyplot as mplt


if sys.argv[1] == 'test':
	df=pnd.read_csv("train_track.csv")
	x1=nmpy.arange(len(df))
	y11=nmpy.array(df['SVM_class'])
	y22=nmpy.array(df['NB_Class'])
	y33=nmpy.array(df['RF_class'])

	mplt.plot(x1,y11)
	mplt.plot(x1,y22)
	mplt.plot(x1,y33)

	mplt.xlabel('batches ->')  
	mplt.ylabel('Score ->')

	mplt.legend(['SVM', 'Naive Bias', 'Random Forest'])

	mplt.show()

else:
	df2= pnd.read_csv("cluster_track.csv")
	x1=nmpy.arange(len(df2))
	y11=nmpy.array(df2['centroid1_diff'])
	y22=nmpy.array(df2['centroid2_diff'])
	
	mplt.plot(x1,y11)
	mplt.plot(x1,y22)
	
	mplt.xlabel('batches ->')  
	mplt.ylabel('centroid_shift ->')

	mplt.legend(['centroid_shift 1', 'centroid_shift 2'])

	mplt.show()


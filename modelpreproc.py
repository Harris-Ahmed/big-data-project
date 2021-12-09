import re                                  
import string                             
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer  
import pandas as pnd
import numpy as nmpy
import string
import pickle



from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import csv




stopwords_english = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stemss=PorterStemmer()

def process1(x1):
    x1= x1.lower()
    x1 = re.sub(r'^RT[\s]+', '.' , x1)
    x1 = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',x1)
    x1 = re.sub(r'#', '' , x1)
    x1 = re.sub(r'[0-9]', '.' , x1)
    x1 = re.sub(r'(\\u[0-9A-Fa-f]+)', '.', x1)       
    x1 = re.sub(r'[^\x00-\x7f]' , '.',x1)
    x1 = re.sub('@[^\s]+','atUser',x1)
    x1 = re.sub(r"(\!)\1+", ' multiExclamation', x1)
    x1 = re.sub(r"(\?)\1+", ' multiQuestion', x1)
    x1 = re.sub(r"(\.)\1+", ' multistop', x1)
    return x1

def tokens(x1):
   
    return x1.split()

def process2(x1):
    res=[]
    for wrds in x1:
        if wrds not in stopwords_english and wrds not in string.punctuation:
            res.append(wrds)
    return res

def process3(x1):
    res=[]
    for wrds in x1:
        stem_word= stemss.stem(wrds)
        res.append(stem_word)
    return " ".join(res)
    
    


def preprocess(df):
    df['feature1']= df['feature1'].apply(lambda x1: process1(x1))
    df['feature1']= df['feature1'].apply(lambda x1: tokens(x1))
    df['feature1']= df['feature1'].apply(lambda x1: process2(x1))
    df['feature1']= df['feature1'].apply(lambda x1: process3(x1))
    return df



print("####### processed ######")

	

def svm_classifier(X_train, X_test, y_train, y_test):
	SVCmodel = svm.LinearSVC()
	SVCmodel.fit(X_train, y_train)
	y_predic2 = SVCmodel.predict(X_test)
	file1 = 'final1_model.sav'
	pickle.dump(SVCmodel, open(file1, 'wb'))
	return accuracy_score(y_test,y_predic2)

def NB_classifier(X_train, X_test, y_train, y_test):
	clasf = RandomForestClassifier(max_depth=2, random_state=0)
	clasf.fit(X_train,y_train)
	y_predic2= clasf.predict(X_test)
	file2 = 'final2_model.sav'
	pickle.dump(clasf, open(file2, 'wb'))
	return accuracy_score(y_test,y_predic2)
	

def Rf_classifier(X_train, X_test, y_train, y_test):
	clasf=BernoulliNB()
	clasf.fit(X_train,y_train)
	y_predic2=clasf.predict(X_test)
	file3 = 'final3_model.sav'
	pickle.dump(clasf, open(file3, 'wb'))
	return accuracy_score(y_test,y_predic2)
	


def test_cycle(x1,y):
	svm_model = pickle.load(open('final1_model.sav', 'rb'))
	nb_model = pickle.load(open('final2_model.sav','rb'))
	rf_model = pickle.load(open('final3_model.sav','rb'))
	svm_model.fit(x1,y)
	nb_model.fit(x1,y)
	rf_model.fit(x1,y)
	svm_pred= svm_model.predict(x1)
	nb_pred = nb_model.predict(x1)
	rf_pred = rf_model.predict(x1)
	print("### report of SVM ###")
	res1=classification_report(y, svm_pred)
	print(res1)
	print("")
	print("### report of NB_classifier ###")
	print(classification_report(y, nb_pred))
	print("")
	print("### report of RF_classifier ###")
	print(classification_report(y, rf_pred))
	print("")


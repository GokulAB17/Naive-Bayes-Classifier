#1) Prepare a classification model using Naive Bayes 
#for salary data 
#Data Description:
#age -- age of a person
#workclass	-- A work class is a grouping of work 
#education	-- Education of an individuals	
#maritalstatus -- Marital status of an individulas	
#occupation	 -- occupation of an individuals
#relationship -- 	
#race --  Race of an Individual
#sex --  Gender of an Individual
#capitalgain --  profit received from the sale of an investment	
#capitalloss	-- A decrease in the value of a capital asset
#hoursperweek -- number of hours work per week	
#native -- Native of an individual
#Salary -- salary of an individual


#Importing required packages in the environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

#Loading Dataset boh training and test
salary_train = pd.read_csv(r"filepath\SalaryData_Train.csv")
salary_test = pd.read_csv(r"filepath\SalaryData_Test.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

#Preprocessing Data by proving labels to categorical data
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])


colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

#Applying Naive Bayes Classification on test dataset
sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 80%

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print ("Accuracy",(10891+780)/(10891+780+2920+469))
#77.49%


spred_gnb=number.fit_transform(spred_gnb)
y_val=number.fit_transform(testY)
spred_mnb=number.fit_transform(spred_gnb)

#for GaussianNB() analysis
# ROC curve 
from sklearn import metrics
help("metrics")
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(y_val,spred_gnb)
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
#auc=0.636
#Classification report
from sklearn.metrics import classification_report
report1=classification_report(y_val,spred_gnb)

#for MultinomialNB analysis 
#ROC curve
fpr, tpr, threshold = metrics.roc_curve(y_val,spred_mnb)
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
report2=classification_report(y_val,spred_gnb)

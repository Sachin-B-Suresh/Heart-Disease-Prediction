
# coding: utf-8

# In[160]:


import pandas as pd
import numpy as np
from statistics import mean,median,mode
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
train = pd.read_csv('HeartDisease_Training Data.csv')
test = pd.read_csv('HeartDisease_Testing Data.csv')

import cgitb
import cgi
#import MySQLdb
#import Tkinter
#import tkMessageBox

#db = MySQLdb.connect(host="localhost",user="root",passwd="",db="heart")
#cur = db.cursor()

# In[161]:


target = train['Class']
data = train.drop(['Class'],axis=1)


# In[162]:


clf = DecisionTreeClassifier()
clf = clf.fit(data,target)


# In[163]:


prediction = clf.predict(test.drop(['uclass'],axis = 1))
true = [0 if i == 'Patient doesnt have a heart disease' else 1 for i in np.array(test['uclass'])]
print("Model Accuracy : ",accuracy_score(true, prediction))


# In[166]:


bp = int(input('Enter Blood pressure:'))
cholestrol = float(input('Enter cholestrol:'))
fh = int(input('Family history 1- Yes, 0-No:'))
bmi = int(input('Enter BMI:'))
age = int(input('Enter Age:'))

prediction = clf.predict(np.array([bp,cholestrol,fh,bmi,age]).reshape(1,-1))
"""if prediction[0] == 0:
	tkMessageBox.showinfo("Prediction", "Negative: Does not have a heart disease")
else:
	tkMessageBox.showinfo("Prediction", "Positive: Probably Has a heart disease")"""
print("Negative: Does not have a heart disease" if prediction[0] == 0 else "Positive: Probably Has a heart disease")



# In[167]:


export_graphviz(clf,out_file='tree.dot')#exports a Decision tree figure
pos = train[train['Class'] == 1].drop(['Class'],axis = 1)#people with a heart disease
neg = train[train['Class'] == 0].drop(['Class'],axis = 1)#people without a heart disease


# In[183]:


fig, axs = plt.subplots(1, 2)
axs[0].hist(pos['Systolic Blood Pressure'],label="Plot of Blood Pressure of people with heart disease")
axs[1].hist(neg['Systolic Blood Pressure'],label="Plot of Blood Pressure of people no with heart disease")
plt.legend()
print('**Blood Pressure of people with heart disease**')
print('Mean BP:',mean(pos['Systolic Blood Pressure']))
#meanbp=mean(pos['Systolic Blood Pressure'])
print('Median BP:',median(pos['Systolic Blood Pressure']))
#maxbp=median(pos['Systolic Blood Pressure'])
print("**Blood Pressure of people no with heart disease**")
print('Mean:',mean(neg['Systolic Blood Pressure']))
#meanNegbp=mean(neg['Systolic Blood Pressure'])
print('Median:',median(neg['Systolic Blood Pressure']))
#maxNegbp=median(neg['Systolic Blood Pressure'])
plt.show()
#cur.execute("insert into details values(%s,%s,%s,%s)",(mean(pos['Systolic Blood Pressure']),median(pos['Systolic Blood Pressure']),mean(neg['Systolic Blood Pressure']),median(neg['Systolic Blood Pressure'])))

# In[185]:


fig, axs = plt.subplots(1, 2)
axs[0].hist(pos['Age'],label="Plot of ages of people with heart disease")
axs[1].hist(neg['Age'],label="Plot of ages of people no with heart disease")
plt.legend()
print('**Ages of people with heart disease**')
print('Mean BP:',mean(pos['Age']))
print('Median BP:',median(pos['Age']))
print("**Ages of people no with heart disease**")
print('Mean:',mean(neg['Age']))
print('Median:',median(neg['Age']))
plt.show()


# In[187]:


fig, axs = plt.subplots(1, 2)
axs[0].hist(pos['Cholestrol'],label="Plot of cholestrol levels of people with heart disease")
axs[1].hist(neg['Cholestrol'],label="Plot of cholestrol levels of people with no heart disease")
print("**Cholestrol levels of people with heart disease**")
print('Mean:',mean(pos['Cholestrol']))#cholestrol levels above 5.5 could be a sign of a heart disease
print('Median:',median(pos['Cholestrol']))
print("**Cholestrol levels of people with no heart disease**")
print('Mean:',mean(neg['Cholestrol']))
print('Median:',median(neg['Cholestrol']))
plt.legend()
plt.show()


# In[189]:


fig, axs = plt.subplots(1, 2)
axs[0].hist(pos['BMI'],label="Plot of BMI of people with heart disease")
axs[1].hist(neg['BMI'],label="Plot of BMI of people with no heart disease")
print("**BMI of people with heart disease**")
print('Mean:',mean(pos['BMI']))#BMI above 26.5 could be dangerous
print('Median:',median(pos['BMI']))
print("**BMI of people with no heart disease**")
print('Mean:',mean(neg['BMI']))
print('Median:',median(neg['BMI']))
plt.legend()
plt.show()


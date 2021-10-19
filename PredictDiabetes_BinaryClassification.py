#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pylab import rcParams

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


df=pd.read_csv("pima-indians-diabetes.csv")
df.head()


# In[3]:


#plt.plot(df["TricepsSkinfoldThickness"],df["ClassVariable"],'o')


# In[5]:


#Logistic_Regression binary classification model
#Divide the fram columns
X = df[['NoTimePregnant', 'PlasmaGlucoseConcentration','DiastolicBloodPressure','TricepsSkinfoldThickness','2-HourSerumInsulin','BMI','DiabetesPedigreeFunction','Age']]
y = df['ClassVariable']
#Divide the Data in Tet and Train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
#Train model
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)
#Confusion Matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
import seaborn as sn
sn.heatmap(confusion_matrix, annot=True)
#Accuracy
from sklearn import metrics
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()


# In[9]:


#Naieve Bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
y_nb_pred=nb.predict(X_test)
#Accuracy
print('Accuracy: ',metrics.accuracy_score(y_test, y_nb_pred))
plt.show()


# In[11]:


#K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_knn_pred=knn.predict(X_test)
#Accuracy
print('Accuracy: ',metrics.accuracy_score(y_test, y_knn_pred))
plt.show()


# In[14]:


#RandomForest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=70, oob_score=True,n_jobs=-1,random_state=101,max_features=None,min_samples_leaf=30)
rfc.fit(X_train,y_train)
y_rfc_pred=rfc.predict(X_test)
#Accuracy
print('Accuracy: ',metrics.accuracy_score(y_test, y_rfc_pred))
plt.show()


# In[ ]:





# In[ ]:





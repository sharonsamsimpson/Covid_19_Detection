#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


# In[2]:


df = pd.read_csv('covid.csv')


# In[3]:


X = df.drop(["corona_result"], axis=1)
y = df["corona_result"]


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# In[5]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc.predict(X_test)


# In[6]:


pickle.dump((dtc), open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from joblib import load, dump


# In[5]:


numeric_df = pd.read_csv("processed_genre_numeric_data.csv")


# In[6]:


X_scale_df = numeric_df.drop(columns=["year", "genre_predict", "key_scale", "index"])
Y_scale_df = numeric_df[["key_scale"]]


# In[7]:


X_scale_train, X_scale_test, Y_scale_train, Y_scale_test = train_test_split(X_scale_df, Y_scale_df, test_size=0.2, random_state=42)

LR_scale_clf = LogisticRegression(random_state=0)
LR_scale_clf.fit(X_scale_train, np.array(Y_scale_train["key_scale"]))
LR_Y_scale_pred = LR_scale_clf.predict(X_scale_test)

LR_scale_accuracy = accuracy_score(Y_scale_test, LR_Y_scale_pred)
print(f"Model Accuracy: {LR_scale_accuracy}")


# In[8]:


dump(LR_scale_clf, 'LR_scale.joblib')


# In[ ]:





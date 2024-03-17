#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from joblib import load, dump


# In[8]:


numeric_df = pd.read_csv("processed_genre_numeric_data.csv")


# In[12]:


X_scale_df = numeric_df.drop(columns=["year", "genre_predict", "key_scale", "index"])
Y_scale_df = numeric_df[["key_scale"]]


# In[13]:


print(X_scale_df.shape)


# In[14]:


X_scale_train, X_scale_test, Y_scale_train, Y_scale_test = train_test_split(X_scale_df, Y_scale_df, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
scale_clf = RandomForestClassifier(n_estimators=105, random_state=42)

# Train the model on the training data
scale_clf.fit(X_scale_train, np.array(Y_scale_train["key_scale"]))

# Predict the labels for the test set
Y_scale_pred = scale_clf.predict(X_scale_test)

# Calculate the accuracy of the model
scale_accuracy = accuracy_score(Y_scale_test, Y_scale_pred)
print(f"Model Accuracy: {scale_accuracy}")


# In[18]:


dump(scale_clf, 'RF_scale.joblib')


# In[ ]:





# In[ ]:





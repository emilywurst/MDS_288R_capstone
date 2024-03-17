#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from joblib import load, dump

from sklearn.preprocessing import StandardScaler


# In[6]:


numeric_df = pd.read_csv("processed_genre_numeric_data.csv")
test_idx_df = pd.read_csv("test_indices_RS_42.csv")
train_idx_df = pd.read_csv("train_indices_RS_42.csv")


# In[7]:


test_idx = np.array(test_idx_df).flatten()
train_idx = np.array(train_idx_df).flatten()


# In[8]:


train_df = numeric_df.iloc[train_idx]
test_df = numeric_df.iloc[test_idx]


# In[9]:


train = train_df[~train_df['genre_parse'].isin([4,6,8])]
test = test_df[~test_df['genre_parse'].isin([4,6,8])]


# In[11]:


X_train = np.array(train.drop(columns=["genre_parse", "index", "genre_predict"])).astype('float32')
X_test = np.array(test.drop(columns=["genre_parse",  "index", "genre_predict"])).astype('float32')


# In[12]:


scaler = StandardScaler()
X_RF_train = scaler.fit_transform(X_train)
X_RF_test = scaler.fit_transform(X_test)

Y_RF_train = train[["genre_parse"]]
Y_RF_test = test[["genre_parse"]]


# initialize the Random Forest Classifier
genre_RF_classifier = RandomForestClassifier(n_estimators=150, random_state=42)

# train the model on the training data
genre_RF_classifier.fit(X_RF_train, np.array(Y_RF_train["genre_parse"]))

# predict the labels for the test set
Y_pred = genre_RF_classifier.predict(X_RF_test)

# calculate the accuracy of the model
accuracy = accuracy_score(Y_RF_test, Y_pred)
print(f"Model Accuracy: {accuracy}")


# In[14]:


dump(genre_RF_classifier, 'RF_genre.joblib')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install joblib


# In[11]:


from joblib import load
from joblib import dump

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[17]:


scale_RF_model = "RF_scale.joblib"
test_set_all_features = pd.read_csv("test_set_all_features_RS_42.csv")


# In[13]:


X_scale_test = test_set_all_features.drop(columns=["year", "genre_predict", "key_scale", "index"])
Y_scale_test = test_set_all_features[["key_scale"]]


# In[14]:


def predict(model_file, test_set):
    # Load the model
    model = load(model_file)

    
    # Make predictions
    predictions = model.predict(test_set)
    
    return predictions


# In[15]:


Y_scale_pred = predict(scale_RF_model, X_scale_test)


# In[16]:


scale_accuracy = accuracy_score(Y_scale_test, Y_scale_pred)
print(f"Model Accuracy: {scale_accuracy}")


# In[ ]:





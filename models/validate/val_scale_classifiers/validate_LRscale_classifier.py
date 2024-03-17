#!/usr/bin/env python
# coding: utf-8

# In[2]:


from joblib import load
from joblib import dump

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[7]:


scale_LR_model = "LR_scale.joblib"
test_set_all_features = pd.read_csv("test_set_all_features_RS_42.csv")


# In[8]:


X_scale_test = test_set_all_features.drop(columns=["year", "genre_predict", "key_scale", "index"])
Y_scale_test = test_set_all_features[["key_scale"]]


# In[9]:


def predict(model_file, test_set):
    # Load the model
    model = load(model_file)

    
    # Make predictions
    predictions = model.predict(test_set)
    
    return predictions


# In[10]:


Y_scale_pred = predict(scale_LR_model, X_scale_test)


# In[11]:


scale_accuracy = accuracy_score(Y_scale_test, Y_scale_pred)
print(f"Model Accuracy: {scale_accuracy}")


# In[ ]:





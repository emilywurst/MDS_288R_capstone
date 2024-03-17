#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import load
from joblib import dump

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd


# In[10]:


FNN_model = "FNN_genre_classifier.joblib"
test_df = pd.read_csv("test_set_all_features_RS_42.csv")


# In[11]:


category_mapping = {1: 0, 2: 1, 3: 2, 5: 3, 7: 4, 8:5}
test_df['mapped_category'] = test_df['genre_parse'].map(category_mapping)
Y_test_encoded = pd.get_dummies(test_df['mapped_category'], prefix='category')


# In[12]:


X_test = test_df.drop(columns=["genre_parse", "mapped_category", "index", "genre_predict"]).astype('float32')


# In[19]:


def predict(model_file, test_set):
    # Load the model
    model = load(model_file)

    
    # Make predictions
    predictions = model.predict(test_set)
    
    return predictions


# In[ ]:





# In[20]:


Y_pred = predict(FNN_model, X_test)


# In[23]:


# Evaluate model
test_loss, test_accuracy = load(FNN_model).evaluate(X_test, Y_test_encoded)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[30]:


from joblib import load
from joblib import dump
import zipfile
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

with zipfile.ZipFile('RF_genre.zip', 'r') as zip_ref:
    zip_ref.extractall('')


with zipfile.ZipFile('test_set_all_features_RS_42.zip', 'r') as zip_ref:
    zip_ref.extractall('')
# In[23]:


genre_RF_model = "RF_genre.joblib"
test_df = pd.read_csv("test_set_all_features_RS_42.csv")


# In[24]:


test = test_df[~test_df['genre_parse'].isin([4,6,8])]


# In[25]:


X_test = np.array(test.drop(columns=["genre_parse",  "index", "genre_predict"])).astype('float32')
Y_test = test[["genre_parse"]]


# In[31]:


scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)


# In[26]:


def predict(model_file, test_set):
    # Load the model
    model = load(model_file)

    
    # Make predictions
    predictions = model.predict(test_set)
    
    return predictions


# In[32]:


Y_pred = predict(genre_RF_model,X_test_scaled )


# In[33]:


accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy}")


# In[ ]:





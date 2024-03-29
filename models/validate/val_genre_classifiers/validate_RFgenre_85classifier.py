#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import load
from joblib import dump

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler


# In[10]:


genre_RF_model = "RF_85genre.joblib"
test_df = pd.read_csv("test_set_all_features_RS_42.csv")


# In[11]:


test = test_df[~test_df['genre_parse'].isin([4, 5, 6, 8])]


# In[12]:


X_test = np.array(test.drop(columns=["genre_parse",  "index", "genre_predict"])).astype('float32')


# In[13]:


X_test = np.array(test.drop(columns=["genre_parse",  "index", "genre_predict"])).astype('float32')
Y_test = test[["genre_parse"]]


# In[14]:


scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)


# In[15]:


def predict(model_file, test_set):
    # Load the model
    model = load(model_file)

    
    # Make predictions
    predictions = model.predict(test_set)
    
    return predictions


# In[16]:


Y_pred = predict(genre_RF_model,X_test_scaled )


# In[17]:


accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy}")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd

from joblib import load, dump


# In[4]:


numeric_df = pd.read_csv("processed_genre_numeric_data.csv")


# In[18]:


test_idx_df = pd.read_csv("test_indices_RS_42.csv")
train_idx_df = pd.read_csv("train_indices_RS_42.csv")


# In[19]:


test_idx = np.array(test_idx_df).flatten()
train_idx = np.array(train_idx_df).flatten()


# In[20]:


category_mapping = {1: 0, 2: 1, 3: 2, 5: 3, 7: 4, 8:5}

train_df = numeric_df.iloc[train_idx]
test_df = numeric_df.iloc[test_idx]

# Apply the mapping
train_df['mapped_category'] = train_df['genre_parse'].map(category_mapping)
test_df['mapped_category'] = test_df['genre_parse'].map(category_mapping)

# Perform one-hot encoding
Y_train_encoded = pd.get_dummies(train_df['mapped_category'], prefix='category')
Y_test_encoded = pd.get_dummies(test_df['mapped_category'], prefix='category')


# In[21]:


X_train = np.array(train_df.drop(columns=["genre_parse", "mapped_category", "index", "genre_predict"])).astype('float32')
X_test = test_df.drop(columns=["genre_parse", "mapped_category", "index", "genre_predict"]).astype('float32')


# In[22]:


# Define FNN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'), 
    Dense(6, activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train_encoded, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, Y_test_encoded)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")


# In[24]:


dump(model, "FNN_genre_classifier.joblib")


# In[ ]:





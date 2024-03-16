#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, OrdinalEncoder
#rename directory to where whole data file is located
directory = 'data.csv'
df = pd.read_csv(directory)
df = df.drop(['Unnamed: 0','submission_offset','submission_offset_2','submission_offset_2.1'], axis = 1)
df['year'] = pd.to_numeric(df['year'], errors = 'coerce')
df.drop_duplicates(subset = ['song','artist','album'],inplace = True)
lis = []
for d in df['true_genre']:
    if type(d) == str:
        d = d.lower()
        g = ''
        if 'jazz' in d:
            g = g + 'jazz/'
        if 'electronic' in d:
            g = g + 'electronic/'
        if ('folk' in d) or ('country' in d):
            g = g + 'folkcountry/'
        if ('funk' in d) or ('soul' in d) or ('rnb' in d) or ('r&b' in d):
            g = g + 'funksoulrnb/'
        if 'pop' in d:
            g = g + 'pop/'
        if ('rap' in d) or (('hip' in d) and ('hop' in d)):
            g = g + 'rap/'
        if ('classical' in d) or ('baroque' in d):
            g = g + 'classical/'
        if 'alternative' in d:
            g = g + 'alternative/'
        if 'blues' in d:
            g = g+ 'blues/'
        if 'rock' in d:
            g = g + "rock"
        if g == '':
            lis.append(None)
        else:
            lis.append(g.strip("/"))
    else:
        lis.append(None)        
df['genre_parse'] = lis
genre = df[df['genre_parse'].isnull() == False]
genre1 = genre[~genre['genre_parse'].str.contains('/')]
enc = OrdinalEncoder()
arr = enc.fit_transform(genre1[['key_scale','key_key']])
genre1['key_scale'],genre1['key_key'] = arr[:,0],arr[:,1]
genre1 = genre1.dropna(subset = 'year',ignore_index=True)
genre1.to_csv('year_prediction_base_data.csv',index=False)


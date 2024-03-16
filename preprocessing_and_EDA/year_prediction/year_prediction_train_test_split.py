#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, OrdinalEncoder
directory = 'year_prediction_base_data.csv'
genre1 = pd.read_csv(directory)
#separates into before 1985 and after
gy = genre1[(genre1['year']<= 1985) & (genre1['year']>=1950)]
gy1 = genre1[(genre1['year']<= 2015) & (genre1['year']>=1986)]
#disproportionate stratified random sampling
test = gy1.groupby('year',group_keys=False).apply(lambda x: x.sample(n = 4333,random_state = 42))
#puts all together and random order
gy2 = pd.concat([gy,test])
gy2 = gy2.sample(frac=1,random_state = 30)
#separates x and y and creates test set
gy2.loc[gy2['year']<1960,'decade'] = '50s'
gy2.loc[(gy2['year']<1970)&(gy2['year']>1959),'decade'] = '60s'
gy2.loc[(gy2['year']<1980)&(gy2['year']>1969),'decade'] = '70s'
gy2.loc[(gy2['year']<1990)&(gy2['year']>1979),'decade'] = '80s'
gy2.loc[(gy2['year']<2000)&(gy2['year']>1989),'decade'] = '90s'
gy2.loc[(gy2['year']<2010)&(gy2['year']>1999),'decade'] = '00s'
gy2.loc[(gy2['year']<2020)&(gy2['year']>2009),'decade'] = '10s'
decade = gy2['decade']
gy_y = gy2['year']
gy_x = gy2.drop(['year','decade'],axis = 1)
gy_x = normalize(gy_x.select_dtypes(include = 'number'), norm = 'max')
gy_xte = gy_x[200000:]
gy_yte = gy_y[200000:]
gy_yte2 = decade[200000:]
#drops test values from dataset
genre3 = genre1.drop(list(gy_yte.index))

gy = genre3[(genre3['year']<= 1985) & (genre3['year']>=1950)]
gy1 = genre3[(genre3['year']<= 2015) & (genre3['year']>=1986)]
test = gy1.groupby('year',group_keys=False).apply(lambda x: x.sample(n = 3333,random_state = 0,replace = True))
#print("Length 1950-1985:",str(len(gy)))
#print("Length 1986-2015:",str(len(test)))

# gy1 = gy1.sample(n = 131921, replace = False, random_state = 42)
gy2 = pd.concat([gy,test])
gy2 = gy2.sample(frac=1, random_state = 50)
gy2.loc[gy2['year']<1960,'decade'] = '50s'
gy2.loc[(gy2['year']<1970)&(gy2['year']>1959),'decade'] = '60s'
gy2.loc[(gy2['year']<1980)&(gy2['year']>1969),'decade'] = '70s'
gy2.loc[(gy2['year']<1990)&(gy2['year']>1979),'decade'] = '80s'
gy2.loc[(gy2['year']<2000)&(gy2['year']>1989),'decade'] = '90s'
gy2.loc[(gy2['year']<2010)&(gy2['year']>1999),'decade'] = '00s'
gy2.loc[(gy2['year']<2020)&(gy2['year']>2009),'decade'] = '10s'
gy_y = gy2['year']
gy_ydec = gy2['decade']
gy_x = gy2.drop(['year','decade'],axis = 1)
gy_x = normalize(gy_x.select_dtypes(include = 'number'), norm = 'max')

gy_xte = pd.DataFrame(data = gy_xte, 
                      columns = genre1.select_dtypes(include = 'number').drop('year',axis = 1).columns)

gy_x = pd.DataFrame(data = gy_x, 
                      columns = genre1.select_dtypes(include = 'number').drop('year',axis = 1).columns)

gy_xte.to_csv('year_prediction_test_x_data.csv',index=False)
gy_yte.to_csv('year_prediction_test_y_data.csv',index=False)
gy_yte2.to_csv('decade_prediction_test_y_data.csv', index = False)
gy_x.to_csv('year_prediction_train_x_data.csv',index=False)
gy_y.to_csv('year_prediction_train_y_data.csv',index=False)
gy_ydec.to_csv('decade_prediction_train_y_data.csv',index=False)


#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import time
notebookstart= time.time()


# In[2]:


import os
#from typing import Tuple

import pandas as pd
#import numpy as np


# In[ ]:





# In[3]:


DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# In[4]:


get_ipython().system('pip install catboost')


# In[ ]:





# In[5]:


from catboost import CatBoostRegressor
from catboost import Pool, cv


# In[ ]:





# In[ ]:





# In[6]:


train_df = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))
test_df  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))
train_df.shape, test_df.shape


# In[7]:


train_df.head()


# In[8]:


train_df['class'].value_counts()


# In[ ]:





# In[9]:


get_ipython().run_cell_magic('time', '', 'params = {"iterations": 3500,\n          "loss_function": \'RMSE\',\n          #"loss_function": \'R2\',\n         }\n\ntrain = Pool(data = train_df[[\'x_min\', \'y_min\', \'x_max\', \'y_max\', \'conf\', \'h\', \'w\']],\n             label = train_df[[\'distance\']],\n             #cat_features=cat_features\n            )\n\nscores = cv(train, params,\n            fold_count = 3,\n            verbose = False,\n           )')


# In[10]:


niter = scores['test-RMSE-mean'].argmin() + 13
scores['test-RMSE-mean'].min(), scores['test-RMSE-mean'].argmin(), niter

0.9254814620106862, 444    x/y/conf
0.8717672874064516, 445    x/y/conf/h/w with 0
0.8689944606627914, 608,   x/y/conf/h/w with nulls
# In[ ]:





# In[11]:


get_ipython().run_cell_magic('time', '', "#model_cb = CatBoostRegressor(iterations = niter, verbose = 100)\nmodel_cb = CatBoostRegressor(verbose = 100)\n\n# Fit model\nmodel_cb.fit(train_df[['x_min', 'y_min', 'x_max', 'y_max', 'conf', 'h', 'w']], train_df[['distance']].values)")

999:	learn: 0.2699253	total: 1.49s	remaining: 0us       x/y/conf
999:	learn: 0.2365099	total: 1.23s	remaining: 0us       x/y/conf/h/w with nulls
# In[ ]:





# In[ ]:





# In[12]:


preds = model_cb.predict(test_df[['x_min', 'y_min', 'x_max', 'y_max', 'conf', 'h', 'w']])
test_df['distance'] = preds

sample_solution_df = test_df[['image_name', 'distance']]


# In[13]:


test_img_names = set(os.listdir(DIR_DATA_TEST))


# In[14]:


lost_test_items = []

for file_name in test_img_names - set(sample_solution_df['image_name'].values):
    lost_test_items.append([file_name, 0])
    
lost_test_items_df = pd.DataFrame(lost_test_items, columns=['image_name', 'distance'])
sample_solution_df = pd.concat([sample_solution_df, lost_test_items_df])

sample_solution_df.to_csv(os.path.join(DIR_SUBM, '6_baseline_hw_wnulls.csv'), sep=';', index=False)


# In[ ]:





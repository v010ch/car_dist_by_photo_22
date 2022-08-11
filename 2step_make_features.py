#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import time
notebookstart= time.time()


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import torch


# In[4]:


import os
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener() # for using Image.open for .heic without changes

from tqdm.auto import tqdm
tqdm.pandas()


# In[5]:


from itertools import product
#import multiprocessing as mp
import torch.multiprocessing as mp


# In[6]:


#import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


from src.mkcarfeatures import features as ft


# In[ ]:





# In[ ]:




!pip install multiprocesspandasfrom multiprocesspandas import applyparallel
# In[ ]:





# In[ ]:





# In[9]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_TRAIN = os.path.join(os.getcwd(), 'subm', 'train')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# In[10]:


test_img_names  = set(os.listdir(DIR_DATA_TEST))
train_img_names = set(os.listdir(DIR_DATA_TRAIN))
len(test_img_names), len(train_img_names)


# In[11]:


train_labels_df = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), sep=';', index_col=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


#model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
model.classes = [0, 2]  # person and car
_ = model.cpu()


# In[ ]:





# In[ ]:





# In[13]:


from functools import partial


# In[42]:


get_ipython().run_cell_magic('time', '', 'with mp.Pool(processes = (mp.cpu_count() - 2)) as pool:\n    #results = pool.map(ft.create_feeatures_mp, product(list(train_img_names)[:5], [DIR_DATA_TRAIN], [model]))\n    #results = pool.map(ft.create_feeatures_mp, product(list(train_img_names)[:5], [DIR_DATA_TRAIN]))\n    results = pool.map(partial(ft.create_feeatures_mp, inp_dir = DIR_DATA_TRAIN, inp_model = model), train_img_names)')


# In[41]:


train_df = pd.DataFrame(results, columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class', 'h', 'w'])
train_df = pd.merge(train_labels_df, train_df, how='left')
train_df.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




train_data = []

for img_name in tqdm(train_img_names): 
    #if 'heic' in img_name:
    #    heif_file = pyheif.read(os.path.join(DIR_DATA_TRAIN, img_name))
    #   img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride)
    #else:
    #    img = Image.open(os.path.join(DIR_DATA_TRAIN, img_name))
    img = Image.open(os.path.join(DIR_DATA_TRAIN, img_name))
    
    #img_ = np.array(img)
    results = model(np.array(img))
    
    if results.xyxy[0].shape != torch.Size([0, 6]):
        
        #img_cntr = (int(img_.shape[1]/2), int(img_.shape[0]/2))
        #target_goal = determine_targ_car(results, img_cntr)
        target_goal = 0
        
        
        results = [img_name] + results.xyxy[0][target_goal].numpy().tolist()
        train_data.append(results)#train_df = ft.create_feeatures(list(train_img_names)[:10], DIR_DATA_TRAIN, model, use_centr = True) #use_centr
train_df = ft.create_feeatures(train_img_names, DIR_DATA_TRAIN, model, use_centr = True) #use_centr
train_df = pd.merge(train_labels_df, train_df, how='left')
train_df.shapetrain_df.columns
# In[43]:


train_df[train_df.image_name == 'img_1890.jpg']


# In[ ]:





# In[ ]:





# In[44]:


get_ipython().run_cell_magic('time', '', 'with mp.Pool(processes = (mp.cpu_count() - 2)) as pool:\n    #results = pool.map(ft.create_feeatures_mp, product(list(train_img_names)[:5], [DIR_DATA_TRAIN], [model]))\n    #results = pool.map(ft.create_feeatures_mp, product(list(train_img_names)[:5], [DIR_DATA_TRAIN]))\n    results = pool.map(partial(ft.create_feeatures_mp, inp_dir = DIR_DATA_TEST, inp_model = model), test_img_names)')


# In[46]:


test_df = pd.DataFrame(results, columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class', 'h', 'w'])
test_df.shape

test_df = create_feeatures(test_img_names, DIR_DATA_TEST, model, use_centr = True) #use_centr
test_df.shape
# In[55]:


#test_df[test_df.image_name == 'img_2571.jpg']
#test_df.head()


# yolov5 не найдено машин:   
# train: img_1890.jpg,    
# test: img_1888.jpg, img_2674.heic, img_2571.jpg, img_1889.jpg(only person)   

# In[ ]:





# In[56]:


sns.histplot(train_df, x='h')
plt.show()


# In[57]:


sns.histplot(train_df, x='w')
plt.show()


# In[ ]:





# In[58]:


train_df['class'].value_counts()


# In[59]:


test_df['class'].value_counts()

2.0    525
0.0      42.0    514
0.0      4
# In[ ]:





# In[60]:


for el in ['x_min', 'y_min', 'x_max', 'y_max', 'h', 'w']:
    train_df[f'log_{el}'] = train_df[el].apply(lambda x: np.log(x))
    test_df[f'log_{el}'] = test_df[el].apply(lambda x: np.log(x))


# In[ ]:





# In[61]:


train_df.head(20)


# In[ ]:





# In[62]:


train_df.to_csv(os.path.join(DIR_DATA, 'train_upd.csv'), index = False)
test_df.to_csv(os.path.join(DIR_DATA, 'test_upd.csv'), index = False)


# In[ ]:





# In[ ]:





# In[ ]:





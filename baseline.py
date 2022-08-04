#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pyheif


# In[2]:


get_ipython().system('pip install catboost')


# In[ ]:





# In[3]:


import pandas as pd
import os
from PIL import Image
import numpy as np
#import pyheif 
#from tqdm.notebook import tqdm
from tqdm.auto import tqdm
tqdm.pandas()


# In[4]:


from pillow_heif import register_heif_opener

register_heif_opener()


# In[32]:


DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[6]:


test_img_names = set(os.listdir(DIR_DATA_TEST))
train_img_names = set(os.listdir(DIR_DATA_TRAIN))

#test_img_names = set(os.listdir('NordClan/participants/test'))
#train_img_names = set(os.listdir('NordClan/participants/train'))


# In[7]:


print(test_img_names.intersection(train_img_names))


# In[8]:


train_labels_df = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), sep=';', index_col=None)


# In[9]:


train_labels_names = set(train_labels_df['image_name'].values)


# In[10]:


train_labels_names.intersection(test_img_names)


# In[11]:


len(train_labels_names.intersection(train_img_names)) == len(train_img_names)


# In[12]:


train_labels_df['image_name'].value_counts().head(5)


# In[13]:


img_name = 'img_1596' + '.jpg'
train_labels_df[train_labels_df['image_name'] == img_name]


# In[14]:


img = Image.open(os.path.join(DIR_DATA_TRAIN, img_name))
img.thumbnail((640, 640), Image.ANTIALIAS)
img


# ## Train / test

# In[15]:


import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0, 2]  # person and car


# In[16]:


_ = model.cpu()


# In[17]:


train_data = []

for img_name in tqdm(train_img_names): 
    #if 'heic' in img_name:
    #    heif_file = pyheif.read(os.path.join(DIR_DATA_TRAIN, img_name))
    #   img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride)
    #else:
    #    img = Image.open(os.path.join(DIR_DATA_TRAIN, img_name))
    img = Image.open(os.path.join(DIR_DATA_TRAIN, img_name))
    
    results = model(np.array(img))
    #results.tocpu()
    if results.xyxy[0].shape != torch.Size([0, 6]):
        results = [img_name] + results.xyxy[0][0].numpy().tolist()
        train_data.append(results)


# In[18]:


train_data_df = pd.DataFrame(train_data, columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class'])


# In[19]:


train_data_df = pd.merge(train_labels_df, train_data_df, how='left')


# In[21]:


from catboost import CatBoostRegressor


# In[22]:


get_ipython().run_cell_magic('time', '', "model_2 = CatBoostRegressor()\n\n# Fit model\nmodel_2.fit(train_data_df[['x_min', 'y_min', 'x_max', 'y_max', 'conf']], train_data_df[['distance']].values)")


# In[24]:


test_data = []

for img_name in tqdm(test_img_names): 
    #if 'heic' in img_name:
    #    heif_file = pyheif.read(os.path.join('NordClan/participants/test/', img_name))
    #    img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride)
    #else:
    #    img = Image.open(os.path.join('NordClan/participants/test/', img_name))
    img = Image.open(os.path.join(DIR_DATA_TEST, img_name))
    results = model(np.array(img))
    if results.xyxy[0].shape != torch.Size([0, 6]):
        results = [img_name] + results.xyxy[0][0].numpy().tolist()
        test_data.append(results)


# In[25]:


test_data_df = pd.DataFrame(test_data, columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class'])


# In[26]:


preds = model_2.predict(test_data_df[['x_min', 'y_min', 'x_max', 'y_max', 'conf']])


# In[27]:


test_data_df['distance'] = preds


# In[28]:


sample_solution_df = test_data_df[['image_name', 'distance']]


# In[29]:


lost_test_items = []

for file_name in test_img_names - set(sample_solution_df['image_name'].values):
    lost_test_items.append([file_name, 0])


# In[30]:


lost_test_items_df = pd.DataFrame(lost_test_items, columns=['image_name', 'distance'])


# In[31]:


sample_solution_df = pd.concat([sample_solution_df, lost_test_items_df])


# In[33]:


sample_solution_df.to_csv(os.path.join(DIR_SUBM, 'baseline.csv'), sep=';', index=False)


# In[ ]:





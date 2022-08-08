#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import time
notebookstart= time.time()


# In[2]:


import torch


# In[3]:


import os
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener() # for using Image.open for .heic without changes

from tqdm.auto import tqdm
tqdm.pandas()


# In[4]:


#import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[6]:


DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# In[7]:


test_img_names  = set(os.listdir(DIR_DATA_TEST))
train_img_names = set(os.listdir(DIR_DATA_TRAIN))
len(test_img_names), len(train_img_names)


# In[8]:


train_labels_df = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), sep=';', index_col=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


def get_car_center(inp_tensor: torch.Tensor) -> Tuple[int, int]:

    car_cntr = (int((inp_tensor[2].int().item() - inp_tensor[0].int().item())/2 + inp_tensor[0].int().item()),
                int((inp_tensor[3].int().item() - inp_tensor[1].int().item())/2 + inp_tensor[1].int().item())
        )
    
    return car_cntr


# In[10]:


def get_center_dist(inp_center: Tuple[int, int], inp_point: Tuple[int, int]) -> float:
    
    return np.sqrt((inp_center[0] - inp_point[0])**2 +                    (inp_center[1] - inp_point[1])**2)


# In[11]:


def determine_targ_car(inp_results, inp_img_cntr: Tuple[int, int]) -> int:
    
    min_dist = 1000000
    min_idx = -1
    
    for el in range(inp_results.xyxy[0].shape[0]):
        if inp_results.xyxy[0][el][5].int().item() != 2:
            continue
        car_cntr = get_car_center(inp_results.xyxy[0][el])
        cur_dist = get_center_dist(inp_img_cntr, car_cntr)
        if cur_dist < min_dist:
            min_dist = cur_dist
            min_idx = el

    return min_idx


# In[12]:


def create_feeatures(inp_fnames: List[str], inp_dir: str, inp_model, use_centr: Optional[bool] = False):
    
    ret_data = []

    for img_name in tqdm(inp_fnames): 
        #if 'heic' in img_name:
        #    heif_file = pyheif.read(os.path.join(inp_dir, img_name))
        #   img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride)
        #else:
        #    img = Image.open(os.path.join(inp_dir, img_name))
        img = Image.open(os.path.join(inp_dir, img_name))
        
        
        img = np.array(img)
        #results = model(np.array(img))
        results = model(img)
    
        if results.xyxy[0].shape != torch.Size([0, 6]):

            if use_centr:
                #img_cntr = (int(img_.shape[1]/2), int(img_.shape[0]/2))
                img_cntr = (int(img.shape[1]/2), int(img.shape[0]/2))
                target_goal = determine_targ_car(results, img_cntr)
            else:
                target_goal = 0

            if target_goal < 0:
                print(f'wtf2, {img_name}   {results.xyxy[0].shape}')
                continue
                
            h = results.xyxy[0][target_goal][3] - results.xyxy[0][target_goal][1]
            w = results.xyxy[0][target_goal][2] - results.xyxy[0][target_goal][0]
            results = results.xyxy[0][target_goal].numpy().tolist() + [h.item(), w.item()]
            
            # позволим алгоритмам самим выбирать как заполнить пропуски
            ret_data.append([img_name] + results)
            
            
        else:
            print(f'wtf, {img_name}   {results.xyxy[0].shape}')
            # позволим алгоритмам самим выбирать как заполнить пропуски
            #results = [0, 0, 0, 0, 0, 0, 0, 0]

# позволим алгоритмам самим выбирать как заполнить пропуски
#        ret_data.append([img_name] + results)
        
    ret_data = pd.DataFrame(ret_data, columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class', 'h', 'w'])
        
    return ret_data


# In[ ]:





# In[13]:


#model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
model.classes = [0, 2]  # person and car
_ = model.cpu()


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
        train_data.append(results)
# In[14]:


train_df = create_feeatures(train_img_names, DIR_DATA_TRAIN, model, use_centr = True) #use_centr
train_df = pd.merge(train_labels_df, train_df, how='left')
train_df.shape


# In[ ]:





# In[15]:


test_df = create_feeatures(test_img_names, DIR_DATA_TEST, model, use_centr = True) #use_centr
test_df.shape


# In[ ]:





# In[16]:


sns.histplot(train_df, x='h')
plt.show()


# In[17]:


sns.histplot(train_df, x='w')
plt.show()


# In[18]:


train_df.head(20)


# In[19]:


train_df['class'].value_counts()


# In[20]:


test_df['class'].value_counts()

2.0    525
0.0      42.0    514
0.0      4
# In[ ]:





# In[21]:


train_df.to_csv(os.path.join(DIR_DATA, 'train_upd.csv'), index = False)
test_df.to_csv(os.path.join(DIR_DATA, 'test_upd.csv'), index = False)


# In[ ]:





# In[ ]:





# In[ ]:





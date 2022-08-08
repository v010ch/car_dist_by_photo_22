#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pyheif


# In[2]:


get_ipython().system('pip install catboost')


# In[3]:


import time
notebookstart= time.time()


# In[4]:


import torch


# In[5]:


import pandas as pd
import os
from PIL import Image
import numpy as np
from typing import Tuple
#import pyheif 
#from tqdm.notebook import tqdm
from tqdm.auto import tqdm
tqdm.pandas()


# In[6]:


from pillow_heif import register_heif_opener
register_heif_opener() # for using Image.open for .heic without changes


# In[7]:


from catboost import CatBoostRegressor
from catboost import Pool, cv


# In[8]:


DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[9]:


test_img_names = set(os.listdir(DIR_DATA_TEST))
train_img_names = set(os.listdir(DIR_DATA_TRAIN))

#test_img_names = set(os.listdir('NordClan/participants/test'))
#train_img_names = set(os.listdir('NordClan/participants/train'))


# In[10]:


print(test_img_names.intersection(train_img_names))


# In[11]:


train_labels_df = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), sep=';', index_col=None)


# In[12]:


train_labels_names = set(train_labels_df['image_name'].values)


# In[13]:


train_labels_names.intersection(test_img_names)


# In[14]:


len(train_labels_names.intersection(train_img_names)) == len(train_img_names)


# In[15]:


train_labels_df['image_name'].value_counts().head(5)


# In[16]:


img_name = 'img_1596' + '.jpg'
train_labels_df[train_labels_df['image_name'] == img_name]


# In[17]:


img = Image.open(os.path.join(DIR_DATA_TRAIN, img_name))
img.thumbnail((640, 640), Image.ANTIALIAS)
img


# ## Train / test

# In[18]:


def get_car_center(inp_tensor: torch.Tensor) -> Tuple[int, int]:
    
    #car_cntr = (int((inp_tensor.xyxy[0][el][2].int().item() - inp_tensor.xyxy[0][el][0].int().item())/2 + inp_tensor.xyxy[0][el][0].int().item()),
    #            int((inp_tensor.xyxy[0][el][3].int().item() - inp_tensor.xyxy[0][el][1].int().item())/2 + inp_tensor.xyxy[0][el][1].int().item())
    #    )

    car_cntr = (int((inp_tensor[2].int().item() - inp_tensor[0].int().item())/2 + inp_tensor[0].int().item()),
                int((inp_tensor[3].int().item() - inp_tensor[1].int().item())/2 + inp_tensor[1].int().item())
        )
    
    return car_cntr


# In[19]:


def get_center_dist(inp_center: Tuple[int, int], inp_point: Tuple[int, int]) -> float:
    
    return np.sqrt((inp_center[0] - inp_point[0])**2 +                    (inp_center[1] - inp_point[1])**2)


# In[20]:


def determine_targ_car(inp_results, inp_img_cntr) -> int:
    
    min = 1000000

    for el in range(inp_results.xyxy[0].shape[0]):
        car_cntr = get_car_center(inp_results.xyxy[0][el])
        cur_dist = get_center_dist(inp_img_cntr, car_cntr)
        if cur_dist < min:
            min = cur_dist
            min_idx = el

    #print(min_idx)
    return min_idx


# In[21]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0, 2]  # person and car
_ = model.cpu()


# In[1]:


train_data = []

for img_name in tqdm(train_img_names): 
    #if 'heic' in img_name:
    #    heif_file = pyheif.read(os.path.join(DIR_DATA_TRAIN, img_name))
    #   img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride)
    #else:
    #    img = Image.open(os.path.join(DIR_DATA_TRAIN, img_name))
    img = Image.open(os.path.join(DIR_DATA_TRAIN, img_name))
    
    img_ = np.array(img)
    results = model(np.array(img))
    #results.tocpu()
    if results.xyxy[0].shape != torch.Size([0, 6]):
        
        #img_cntr = (int(img_.shape[1]/2), int(img_.shape[0]/2))
        #target_goal = determine_targ_car(results, img_cntr)
        target_goal = 0
        
        results = [img_name] + results.xyxy[0][target_goal].numpy().tolist()
        train_data.append(results)


# In[23]:


train_data_df = pd.DataFrame(train_data, columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class'])


# In[24]:


train_data_df = pd.merge(train_labels_df, train_data_df, how='left')


# In[25]:


train_data_df.shape


# In[26]:


motion_blur_train = ['img_2709.heic', 'img_2733.heic', 'img_2734.heic'] 
for el in motion_blur_train:
    idx = train_data_df[train_data_df.image_name == el].index.values[0]
    train_data_df.drop([idx], inplace = True)    


# In[27]:


train_data_df.shape


# In[ ]:





# ## Моделирование

# In[41]:


get_ipython().run_cell_magic('time', '', 'params = {"iterations": 3500,\n          "loss_function": \'RMSE\',\n          #"loss_function": \'R2\',\n         }\n\ntrain = Pool(data = train_data_df[[\'x_min\', \'y_min\', \'x_max\', \'y_max\', \'conf\']],\n             label = train_data_df[[\'distance\']],\n             #cat_features=cat_features\n            )\n\nscores = cv(train, params,\n            fold_count = 3,\n            verbose = False,\n           )')


# In[29]:


niter = scores['test-RMSE-mean'].argmin() + 13
scores['test-RMSE-mean'].min(), scores['test-RMSE-mean'].argmin(), niter


# In[30]:


get_ipython().run_cell_magic('time', '', "#model_cb = CatBoostRegressor(iterations = niter, verbose = 100)\nmodel_cb = CatBoostRegressor(verbose = 100)\n\n# Fit model\nmodel_cb.fit(train_data_df[['x_min', 'y_min', 'x_max', 'y_max', 'conf']], train_data_df[['distance']].values)")


# In[ ]:





# In[31]:


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


# In[32]:


test_data_df = pd.DataFrame(test_data, columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class'])


# In[33]:


preds = model_cb.predict(test_data_df[['x_min', 'y_min', 'x_max', 'y_max', 'conf']])


# In[34]:


test_data_df['distance'] = preds


# In[35]:


sample_solution_df = test_data_df[['image_name', 'distance']]


# In[36]:


lost_test_items = []

for file_name in test_img_names - set(sample_solution_df['image_name'].values):
    lost_test_items.append([file_name, 0])


# In[37]:


lost_test_items_df = pd.DataFrame(lost_test_items, columns=['image_name', 'distance'])


# In[38]:


sample_solution_df = pd.concat([sample_solution_df, lost_test_items_df])


# In[39]:


sample_solution_df.to_csv(os.path.join(DIR_SUBM, '4_baseline_drop_motion_blur.csv'), sep=';', index=False)


# In[ ]:





# In[40]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '')


# In[2]:


import time
notebookstart= time.time()


# In[ ]:





# In[3]:


import torch
import os
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener() # for using Image.open for .heic without changes

from tqdm.auto import tqdm
tqdm.pandas()


# In[15]:


import cv2


# In[4]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





# In[ ]:





# In[5]:


# seed the RNG for all devices (both CPU and CUDA)
#torch.manual_seed(1984)

#Disabling the benchmarking feature causes cuDNN to deterministically select an algorithm, 
#possibly at the cost of reduced performance.
#torch.backends.cudnn.benchmark = False

# for custom operators,
import random
random.seed(5986721)

# 
np.random.seed(62185)

#sklearn take seed from a line abowe

CB_RANDOMSEED  = 309487
XGB_RANDOMSEED = 56
LGB_RANDOMSEED = 874256


# In[ ]:





# In[6]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_TRAIN = os.path.join(os.getcwd(), 'subm', 'train')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


DIR_DATA_TRAIN_LP = os.path.join(DIR_DATA, 'train', 'lp')


# In[ ]:





# In[ ]:





# In[7]:


def get_car_center(inp_tensor: torch.Tensor) -> Tuple[int, int]:

    car_cntr = (int((inp_tensor[2].int().item() - inp_tensor[0].int().item())/2 + inp_tensor[0].int().item()),
                int((inp_tensor[3].int().item() - inp_tensor[1].int().item())/2 + inp_tensor[1].int().item())
        )
    
    return car_cntr


# In[8]:


def get_center_dist(inp_center: Tuple[int, int], inp_point: Tuple[int, int]) -> float:
    
    return np.sqrt((inp_center[0] - inp_point[0])**2 +                    (inp_center[1] - inp_point[1])**2)


# In[9]:


def determine_targ_car(inp_results, inp_img_cntr: Tuple[int, int]) -> int:
    
    min_dist = 1000000
    min_idx  = -1
    
    for el in range(inp_results.xyxy[0].shape[0]):
        # учитываем только машины
        if inp_results.xyxy[0][el][5].int().item() != 2:
            continue
            
        # минимальные габариты учитываемых машин
        # в противном случае иногда ближе к центру оказываются машины например 27х54
        h = inp_results.xyxy[0][el][3] - inp_results.xyxy[0][el][1]
        w = inp_results.xyxy[0][el][2] - inp_results.xyxy[0][el][0]
        if w < 200 or h < 200:
            continue
            
            
        car_cntr = get_car_center(inp_results.xyxy[0][el])
        cur_dist = get_center_dist(inp_img_cntr, car_cntr)
        if cur_dist < min_dist:
            min_dist = cur_dist
            min_idx = el

    return min_idx


# In[17]:


colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), }


# In[ ]:





# In[11]:


train_list = os.listdir(DIR_DATA_TRAIN)
test_list  = os.listdir(DIR_DATA_TEST)


# In[13]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0, 2]  # person and car

_ = model.cpu()


# In[28]:


#global_car_index = 0
global_car_index = 1500

#for (idx, el) in tqdm(enumerate(train_list[:10])):
#for (idx, el) in tqdm(enumerate(train_list)):
for (idx, el) in tqdm(enumerate(test_list)):
    #img = open_img(os.path.join(DIR_DATA_TRAIN, el))
    img = Image.open(os.path.join(DIR_DATA_TEST, el))
    img = np.array(img)
    
    ims_to_save = img.copy()
    
    results = model(img)
    
    if results.xyxy[0].shape != torch.Size([0, 6]):
        #print(results.xyxy[0].shape)
        for obj in range(results.xyxy[0].shape[0]):
            cv2.rectangle(img, 
                          (results.xyxy[0][obj][0].int().item(), results.xyxy[0][obj][1].int().item()), 
                          (results.xyxy[0][obj][2].int().item(), results.xyxy[0][obj][3].int().item()), 
                          colors[results.xyxy[0][obj][-1].int().item()], 
                          6,
                          #cv2.FILLED
                         )
            #_ = cv2.circle(img, car_cntr, 10, (255, 0, 0), 20)
            #print(obj)

        img_cntr = (int(img.shape[1]/2), int(img.shape[0]/2))
        target_goal = determine_targ_car(results, img_cntr)
        #print(target_goal)
        
        
        
        for obj in range(results.xyxy[0].shape[0]):
            if obj != target_goal:
                sub_img = ims_to_save[results.xyxy[0][obj][1].int().item() : results.xyxy[0][obj][3].int().item(), 
                                      results.xyxy[0][obj][0].int().item() : results.xyxy[0][obj][2].int().item()
                                     ]
                Image.fromarray(sub_img).save(os.path.join(DIR_DATA_TRAIN_LP, f'car_{global_car_index}.jpg'))
                global_car_index += 1
                #np.save(sub_img, os.path.join(DIR_DATA_TRAIN_LP, f'car_{global_car_index}.jpg'))
        
tt = '''
        sub_img = img[results.xyxy[0][target_goal][1].int().item() : results.xyxy[0][target_goal][3].int().item(), 
                      results.xyxy[0][target_goal][0].int().item() : results.xyxy[0][target_goal][2].int().item()
                     ]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        
        img[results.xyxy[0][target_goal][1].int().item() : results.xyxy[0][target_goal][3].int().item(), 
            results.xyxy[0][target_goal][0].int().item() : results.xyxy[0][target_goal][2].int().item()
          ] = res
        
    
    
    cv2.circle(img, img_cntr, 10, (0, 0, 255), 20)
    
    img = cv2.resize(img, [252*4, 252*3])
    #img = cv2.resize(img, [504*4, 504*3])    
               
    cv2.imshow(f'{idx} {el}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    '''
    #break


# In[ ]:





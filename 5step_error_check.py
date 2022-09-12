#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '')


# In[2]:


import time
notebookstart = time.time()


# In[3]:


import os
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

import cv2
#from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener() # for using Image.open for .heic without changes

import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





# # Выставление констант

# In[6]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_TRAIN = os.path.join(os.getcwd(), 'subm', 'train')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# In[ ]:





# In[7]:


def open_img(inp_path: str) -> np.ndarray:
    """
    Открытие изображения с учетом heif формата
    aregs:
        inp_path - путь к изображению
    return:
        np.ndarray - изображение
    """
    if inp_path.endswith('.jpg'):
        ret_img = cv2.imread(inp_path)
    else:
        if pillow_heif.is_supported(inp_path):
            heif_file = pillow_heif.open_heif(inp_path, convert_hdr_to_8bit=False)
            #print("image mode:", heif_file.mode)
            #print("image data length:", len(heif_file.data))
            #print("image data stride:", heif_file.stride)
            if heif_file.has_alpha:
                heif_file.convert_to("BGRA;16")
            else:
                heif_file.convert_to("BGR;16")  # convert 10 bit image to RGB 16 bit.
            #print("image mode:", heif_file.mode)
            ret_img = np.asarray(heif_file)
    
    return ret_img


# In[36]:


def plot_corrc(inp_df: pd.DataFrame, inp_cols: List[str], targ_cols = ['distance']) -> None:
    """
    Отображение корреляций заданных признаков и целевой переменной
    args:
        inp_df - входной датафрейм
        inp_cols  - список входных признаков для отбражения корреляции
        targ_cols - целевая переменная
    return:
    """
    f, ax = plt.subplots(1, 2, figsize=(24, 8))
    sns.heatmap(inp_df[inp_cols + targ_cols].corr(),
                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[0])
    sns.heatmap(inp_df[inp_cols + targ_cols].corr(method = 'spearman'),
                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[1])
    
    sns.pairplot(inp_df[inp_cols + targ_cols], height = 16,
                )
   


# In[ ]:





# # Загружаем данные

# In[9]:


train_df = pd.read_csv(os.path.join(DIR_SUBM_TRAIN, 'train_with_pred.csv'))
train_df.shape


# In[ ]:





# Рассчитаем и посмотрим на ошибки

# In[10]:


train_df['err'] = train_df.distance - train_df.pred
train_df['err_upd'] = train_df.err.apply(lambda x: abs(x))


# In[11]:


train_df.err.hist()


# In[12]:


train_df.err_upd.hist()


# In[13]:


train_df.err.nsmallest(5)


# In[14]:


train_df.err.nlargest(5)


# In[15]:


train_df.sort_values(by ='err_upd', ascending = False, inplace = True)


# In[16]:


train_df.head(20)


# In[39]:


#plot_corrc(train_df, ['err'])


# In[18]:


#plot_corrc(train_df, ['err_upd'])


# In[ ]:





# Посмотрим на кадры с наибольшей (по модулю) ошибкой

# In[34]:


for el in train_df.index[:5]:
    tmp = train_df.loc[el, :]
    
    img = open_img(os.path.join(DIR_DATA_TRAIN, tmp.image_name))

    #  рамка найденного автомобиля
    cv2.rectangle(img, 
                  (int(tmp.car_x_min), int(tmp.car_y_min)), 
                  (int(tmp.car_x_max), int(tmp.car_y_max)),
                  (255, 0, 0), 
                  6,
                 )

    sub_img = img[int(tmp.car_y_min) : int(tmp.car_y_max),
                  int(tmp.car_x_min) : int(tmp.car_x_max)
                 ]

    # рамка номера
    cv2.rectangle(sub_img, 
              (int(tmp.plate_y_min), int(tmp.plate_x_min)), 
              (int(tmp.plate_y_max), int(tmp.plate_x_max)),
              (255, 0, 0), 
              6,
             )
    

    img[int(tmp.car_y_min) : int(tmp.car_y_max),
        int(tmp.car_x_min) : int(tmp.car_x_max)
       ] = sub_img
    
    img = cv2.resize(img, [252*4, 252*3])

    cv2.imshow('bir error', img)
    #cv2.imshow('bir error', sub_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# автомобили определяются верно.    
# автомобильный номер с виду тоже.    

# In[ ]:





# In[22]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import time
notebookstart= time.time()


# In[60]:


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


# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_TRAIN = os.path.join(os.getcwd(), 'subm', 'train')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# In[ ]:





# In[6]:


train_df = pd.read_csv(os.path.join(DIR_SUBM_TRAIN, 'train_with_pred.csv'))
train_df.shape


# In[ ]:





# In[15]:


def open_img(inp_path: str) -> np.ndarray:
    
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


# In[65]:


def plot_corrc(inp_df, inp_cols, targ_cols = ['distance']):
    f, ax = plt.subplots(1, 2, figsize=(24, 8))
    sns.heatmap(inp_df[inp_cols + targ_cols].corr(),
    #sns.heatmap(inp_df.query('c2 == 0')[inp_cols + targ_cols].corr(), \n",
                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[0])
    sns.heatmap(inp_df[inp_cols + targ_cols].corr(method = 'spearman'),
    #sns.heatmap(inp_df.query('c2 == 1')[inp_cols + targ_cols].corr(), \n",
                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[1])
#    sns.heatmap(inp_df.query('c2 == 0')[inp_cols + targ_cols].corr(method = 'spearman'), \n",
#                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[1, 0])\n",
#    sns.heatmap(inp_df.query('c2 == 1')[inp_cols + targ_cols].corr(method = 'spearman'), \n",
#                annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', ax = ax[1, 1])\n",
    if 'distrib_brdr' in inp_df.columns:
        sns.pairplot(inp_df[inp_cols + targ_cols + ['distrib_brdr']], height = 16,
                     hue = 'distrib_brdr', #palette = {\"A\": \"C0\", \"B\": \"C1\"}\n",
                     #markers = ['x', 'o']\n",
                    )
    else:
        sns.pairplot(inp_df[inp_cols + targ_cols], height = 16,
                    )
   


# In[ ]:





# In[ ]:





# Рассчитаем и помотрим на ошибки

# In[33]:


train_df['err'] = train_df.distance - train_df.pred
train_df['err_upd'] = train_df.err.apply(lambda x: abs(x))


# In[45]:


train_df.err.hist()


# In[46]:


train_df.err_upd.hist()


# In[49]:


train_df.err.nsmallest(5)


# In[50]:


train_df.err.nlargest(5)


# In[37]:


train_df.sort_values(by='err_upd', ascending = False, inplace = True)


# In[47]:


train_df.head(20)


# In[66]:


plot_corrc(train_df, ['err'])


# In[62]:


#plot_corrc(train_df, ['err_upd'])


# In[ ]:





# Посмотрим на кадры с наибольней (по модулю) ошибкой

# In[56]:


for el in train_df.index[:5]:
    name, x_min, y_min, x_max, y_max, dist, pred, err = train_df.loc[el, ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'distance', 'pred', 'err']].values
    img = open_img(os.path.join(DIR_DATA_TRAIN, name))

    cv2.rectangle(img, 
                  (int(x_min), int(y_min)), 
                  (int(x_max), int(y_max)),
                  (255, 0, 0), 
                  6,
                  #cv2.FILLED
                 )


    img = cv2.resize(img, [252*4, 252*3])

    cv2.imshow('bir error', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





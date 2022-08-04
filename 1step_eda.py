#!/usr/bin/env python
# coding: utf-8

# In[43]:


import cv2
import pillow_heif

from PIL import Image
from PIL.ExifTags import TAGS


# In[2]:


import os
from collections import Counter

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
tqdm.pandas()


# In[3]:


#import seaborn as sns


# In[4]:


import matplotlib


# In[5]:


#!pip list


# In[ ]:





# In[ ]:





# In[6]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# In[7]:


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


# In[ ]:





# In[ ]:





# ## Загрузка данных

# In[16]:


train_list = os.listdir(DIR_DATA_TRAIN)
test_list  = os.listdir(DIR_DATA_TEST)

train_heic = [el for el in train_list if el.endswith('.heic')]
test_heic  = [el for el in test_list  if el.endswith('.heic')]

train_jpg = [el for el in train_list if el.endswith('.jpg')]
test_jpg  = [el for el in test_list if el.endswith('.jpg')]

fnames_train = set([el.split('.')[0] for el in train_list])
fnames_test  = set([el.split('.')[0] for el in test_list])


# In[9]:


#print(Counter([el.split('.')[1] for el in train_list]))
#print(Counter([el.split('.')[1] for el in test_list]))


# In[ ]:





# In[10]:


train_df = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), delimiter = ';')
train_df.shape


# In[11]:


train_df.head()


# In[12]:


train_df.groupby('image_name').agg('size').value_counts()


# In[ ]:





# In[ ]:





# Посмотрим на размеры

# In[13]:


get_ipython().run_cell_magic('time', '', "#tmp = [fl for fl in train_list if fl.endswith('.jpg')]\n#sizes = [cv2.imread(os.path.join(DIR_DATA_TRAIN, el )).shape \\\nsizes = [open_img(os.path.join(DIR_DATA_TRAIN, el )).shape \\\n         for el in tqdm(train_list)]\nCounter(sizes)")


# In[15]:


get_ipython().run_cell_magic('time', '', "#tmp = [fl for fl in test_list if fl.endswith('.jpg')]\n#sizes = [cv2.imread(os.path.join(DIR_DATA_TEST, el )).shape \\\nsizes = [open_img(os.path.join(DIR_DATA_TEST, el )).shape \\\n         for el in tqdm(test_list)]\nCounter(sizes)")


# In[ ]:





# In[ ]:





# Посмотрим на мин и макс

# In[46]:


train_df.distance.nlargest(5)


# In[45]:


train_df.distance.nsmallest(5)


# In[44]:


print('min ', train_df.distance[train_df.distance.argmin()], '  ', train_df.image_name[train_df.distance.argmin()])
print('max ', train_df.distance[train_df.distance.argmax()], '  ', train_df.image_name[train_df.distance.argmax()])


# In[41]:


#img = open_img(os.path.join(DIR_DATA_TRAIN, train_df.image_name[train_df.distance.argmin()]))
img = open_img(os.path.join(DIR_DATA_TRAIN, train_df.image_name[train_df.distance.argmax()]))
#img = cv2.resize(img, [504*4, 504*3])
img = cv2.resize(img, [252*4, 252*3])
cv2.imshow('random', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[ ]:





# In[17]:





# In[18]:





# In[19]:


train_df['ext'] = train_df.image_name.apply(lambda x: x.split('.')[1])


# In[20]:


train_df.distance.hist(bins = 15)


# In[21]:


# 520 x 112
# 245 x 160
# 290 х 170


# In[ ]:





# Посмотрим пересечение датасетов

# In[23]:


tmp = list(fnames_train.intersection(fnames_test))
len(tmp)


# In[29]:


for idx, el in enumerate(tmp):
    if os.path.exists(os.path.join(DIR_DATA_TRAIN, f'{el}.jpg')):
        img_train = open_img(os.path.join(DIR_DATA_TRAIN, f'{el}.jpg'))
    else:
        img_train = open_img(os.path.join(DIR_DATA_TRAIN, f'{el}.heic'))
        
    if os.path.exists(os.path.join(DIR_DATA_TEST, f'{el}.jpg')):
        img_test = open_img(os.path.join(DIR_DATA_TEST, f'{el}.jpg'))
    else:
        img_test = open_img(os.path.join(DIR_DATA_TEST, f'{el}.heic'))
        
    img_train = cv2.resize(img_train, [252*4, 252*3])
    img_test  = cv2.resize(img_test, [252*4, 252*3])

    img = np.concatenate((img_train, img_test), axis = 1)
    
    cv2.imshow(f'{idx}   {el}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()     
    


# In[ ]:





# Смазанные фото

# In[42]:


#for el in train_heic:
for el in test_heic:
    #img = open_img(os.path.join(DIR_DATA_TRAIN, el))
    img = open_img(os.path.join(DIR_DATA_TEST, el))
    img = cv2.resize(img, [252*4, 252*3])

    cv2.imshow(f'motion blur {el}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


# In[ ]:


motion_blur_train = ['img_2709.heic', 'img_2733.heic', 'img_2733.heic']    # 'img_2733.heic' возможно рабочий 
motion_blur_test  = ['img_2674.heic']


# In[40]:


img = open_img(os.path.join(DIR_DATA_TRAIN, 'img_2734.heic'))
#img = open_img(os.path.join(DIR_DATA_TEST, motion_blur_test[0]))
img = cv2.resize(img, [252*4, 252*3])
               
cv2.imshow('motion blur', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[ ]:





# In[37]:





# Проверим метеданные фотографий

# In[ ]:





# In[44]:


# open the image
image = Image.open(os.path.join(DIR_DATA_TRAIN, train_jpg[0]))
  
# extracting the exif metadata
exifdata = image.getexif()
  
# looping through all the tags present in exifdata
for tagid in exifdata:
      
    # getting the tag name instead of tag id
    tagname = TAGS.get(tagid, tagid)
  
    # passing the tagid to get its respective value
    value = exifdata.get(tagid)
    
    # printing the final result
    print(f"{tagname:25}: {value}")


# In[ ]:





# In[ ]:





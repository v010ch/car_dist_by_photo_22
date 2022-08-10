#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pillow_heif

from PIL import Image
from PIL.ExifTags import TAGS


# In[2]:


import os
from collections import Counter
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
tqdm.pandas()


# In[3]:


#import seaborn as sns


# In[4]:


#import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


#!pip list


# In[ ]:





# In[ ]:





# In[7]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# In[8]:


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


# In[9]:


def plot_corrc(inp_df: pd.DataFrame, inp_cols: List[str], targ_cols: Optional[List[int]] = ['distance']):
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





# In[ ]:





# ## Загрузка данных

# In[10]:


train_list = os.listdir(DIR_DATA_TRAIN)
test_list  = os.listdir(DIR_DATA_TEST)

train_heic = [el for el in train_list if el.endswith('.heic')]
test_heic  = [el for el in test_list  if el.endswith('.heic')]

train_jpg = [el for el in train_list if el.endswith('.jpg')]
test_jpg  = [el for el in test_list if el.endswith('.jpg')]

fnames_train = set([el.split('.')[0] for el in train_list])
fnames_test  = set([el.split('.')[0] for el in test_list])


# In[11]:


#print(Counter([el.split('.')[1] for el in train_list]))
#print(Counter([el.split('.')[1] for el in test_list]))


# In[ ]:





# ### Загружаем данные

# (train_upd загружается только после выполнения 2step_make_features)

# In[12]:


#train_df = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), delimiter = ';')
train_df = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))
test_df  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))
train_df.shape, test_df.shape


# In[13]:


train_df.head()


# In[14]:


train_df.groupby('image_name').agg('size').value_counts()


# In[ ]:





# In[ ]:





# Посмотрим на размеры
%%time
#tmp = [fl for fl in train_list if fl.endswith('.jpg')]
#sizes = [cv2.imread(os.path.join(DIR_DATA_TRAIN, el )).shape \
sizes = [open_img(os.path.join(DIR_DATA_TRAIN, el )).shape \
         for el in tqdm(train_list)]
Counter(sizes)Counter({(3024, 4032, 3): 529, (4032, 3024, 3): 1})%%time
#tmp = [fl for fl in test_list if fl.endswith('.jpg')]
#sizes = [cv2.imread(os.path.join(DIR_DATA_TEST, el )).shape \
sizes = [open_img(os.path.join(DIR_DATA_TEST, el )).shape \
         for el in tqdm(test_list)]
Counter(sizes)Counter({(3024, 4032, 3): 519, (4032, 3024, 3): 2})
# In[ ]:





# Посмотрим на мин и макс

# In[15]:


train_df.distance.nlargest(5)


# In[16]:


train_df.distance.nsmallest(5)


# In[17]:


print('min ', train_df.distance[train_df.distance.argmin()], '  ', train_df.image_name[train_df.distance.argmin()])
print('max ', train_df.distance[train_df.distance.argmax()], '  ', train_df.image_name[train_df.distance.argmax()])


# In[18]:


#img = open_img(os.path.join(DIR_DATA_TRAIN, train_df.image_name[train_df.distance.argmin()]))
img = open_img(os.path.join(DIR_DATA_TRAIN, train_df.image_name[train_df.distance.argmax()]))
#img = cv2.resize(img, [504*4, 504*3])
img = cv2.resize(img, [252*4, 252*3])
cv2.imshow('random', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[ ]:





# In[ ]:





# In[ ]:





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

# In[22]:


tmp = list(fnames_train.intersection(fnames_test))
len(tmp)

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
#for el in train_heic:
for el in test_heic:
    #img = open_img(os.path.join(DIR_DATA_TRAIN, el))
    img = open_img(os.path.join(DIR_DATA_TEST, el))
    img = cv2.resize(img, [252*4, 252*3])

    cv2.imshow(f'motion blur {el}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
# In[23]:


motion_blur_train = ['img_2709.heic', 'img_2733.heic', 'img_2734.heic']    # 'img_2734.heic' возможно рабочий 
motion_blur_test  = ['img_2674.heic']


# In[24]:


img = open_img(os.path.join(DIR_DATA_TRAIN, 'img_2745.heic'))
#img = open_img(os.path.join(DIR_DATA_TEST, motion_blur_test[0]))
img = cv2.resize(img, [252*4, 252*3])
               
cv2.imshow('motion blur', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[ ]:





# In[25]:


train_df.columns


# In[26]:


#train_df['exp_w'] = train_df.w.apply(lambda x: np.exp(x))
train_df['log_w'] = train_df.w.apply(lambda x: np.log(x))


# In[ ]:





# Посмотрим на корреляцию с признаками из train_upd

# In[27]:


#plot_corrc(train_df, ['x_min', 'y_min', 'x_max', 'y_max', 'h', 'w']) #'conf', 
plot_corrc(train_df, ['w', 'log_w'])


# In[ ]:





# In[29]:


train_df.sort_values('w').head(5)


# In[30]:


test_df.sort_values('w').head(5)


# In[32]:


#el = 92 # 394, 313, 314
#name, x_min, y_min, x_max, y_max, dist = train_df.loc[el, ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'distance']].values

el = 122 # 143, 122, 32, 406
name, x_min, y_min, x_max, y_max = test_df.loc[el, ['image_name', 'x_min', 'y_min', 'x_max', 'y_max']].values

img = open_img(os.path.join(DIR_DATA_TEST, name))

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





# In[ ]:





# Проверим метеданные фотографий

# In[ ]:





# In[23]:


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





# In[ ]:





# In[ ]:





# In[24]:


colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), }


# In[25]:


import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0, 2]  # person and car

_ = model.cpu()


# In[26]:


# motion blur img_2733.heic, img_2734.heic
# nearest img_2858.jpg
# several cars img_2896.jpg, img_2885.jpg, !! img_2674.jpg, img_2660.jpg
# part closed img_2723.jpg
# left first img_2694.jpg
# car by side img_2418.jpg, 
# construction on the roof img_1832.jpg
# double car img_1621.jpg


# !!!TEST with strange number

# 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class'


# In[27]:


#img = open_img(os.path.join(DIR_DATA_TRAIN, 'img_2733.heic'))
img = open_img(os.path.join(DIR_DATA_TRAIN, 'img_2674.jpg'))
#results = model(np.array(img))
results = model(img)


# In[28]:


colors[results.xyxy[0][0][-1].int().item()]


# In[29]:


def get_car_center(inp_tensor: torch.Tensor) -> Tuple[int, int]:
    
    #car_cntr = (int((inp_tensor.xyxy[0][el][2].int().item() - inp_tensor.xyxy[0][el][0].int().item())/2 + inp_tensor.xyxy[0][el][0].int().item()),
    #            int((inp_tensor.xyxy[0][el][3].int().item() - inp_tensor.xyxy[0][el][1].int().item())/2 + inp_tensor.xyxy[0][el][1].int().item())
    #    )

    car_cntr = (int((inp_tensor[2].int().item() - inp_tensor[0].int().item())/2 + inp_tensor[0].int().item()),
                int((inp_tensor[3].int().item() - inp_tensor[1].int().item())/2 + inp_tensor[1].int().item())
        )
    
    return car_cntr


# In[30]:


def get_center_dist(inp_center: Tuple[int, int], inp_point: Tuple[int, int]) -> float:
    
    return np.sqrt((inp_center[0] - inp_point[0])**2 +                    (inp_center[1] - inp_point[1])**2)


# In[31]:


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

min = 1000000
#img_cntr = (int(img.shape[0]/2), int(img.shape[1]/2))
img_cntr = (int(img.shape[1]/2), int(img.shape[0]/2))


for el in range(results.xyxy[0].shape[0]):
    car_cntr = get_car_center(results.xyxy[0][el])
    cur_dist = get_center_dist(img_cntr, car_cntr)
    if cur_dist < min:
        min = cur_dist
        min_idx = el
        
print(min_idx)
# In[32]:


img_cntr = (int(img.shape[1]/2), int(img.shape[0]/2))
target_goal = determine_targ_car(results, img_cntr)


# In[33]:


cv2.circle(img, img_cntr, 10, (0, 0, 255), 20)
cv2.rectangle(img, 
              (results.xyxy[0][target_goal][0].int().item(), results.xyxy[0][target_goal][1].int().item()), 
             (results.xyxy[0][target_goal][2].int().item(), results.xyxy[0][target_goal][3].int().item()), 
             colors[results.xyxy[0][target_goal][-1].int().item()], 2)
#image = cv.circle(image, centerOfCircle, radius, color, thickness)
car_cntr = get_car_center(results.xyxy[0][target_goal])
_ = cv2.circle(img, car_cntr, 10, (255, 0, 0), 20)

for el in range(results.xyxy[0].shape[0]):
    cv2.rectangle(img, 
                  (results.xyxy[0][el][0].int().item(), results.xyxy[0][el][1].int().item()), 
                  (results.xyxy[0][el][2].int().item(), results.xyxy[0][el][3].int().item()), 
                  (255,0,0), 2
                 )
    #image = cv.circle(image, centerOfCircle, radius, color, thickness)
    car_cntr = get_car_center(results.xyxy[0][el])
    cv2.circle(img, car_cntr, 10, (255, 0, 0), 20)
    
    text_point = (results.xyxy[0][el][1].int().item(),
                  results.xyxy[0][el][3].int().item()
                 )
                  
    cv2.putText(img, f'car{el}', text_point, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
    
    #break
# In[34]:


img = cv2.resize(img, [252*4, 252*3])
               
cv2.imshow('motion blur', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[ ]:





# In[ ]:





# In[44]:


#for idx, el in enumerate(train_list[:10]):
for idx, el in enumerate(train_list[:10]):
    img = open_img(os.path.join(DIR_DATA_TRAIN, el))
    
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
        sub_img = img[results.xyxy[0][target_goal][0].int().item() : results.xyxy[0][target_goal][1].int().item(), 
                      results.xyxy[0][target_goal][2].int().item() : results.xyxy[0][target_goal][3].int().item()
                     ]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        
        img[results.xyxy[0][target_goal][0].int().item() : results.xyxy[0][target_goal][1].int().item(), 
            results.xyxy[0][target_goal][2].int().item() : results.xyxy[0][target_goal][3].int().item()
           ] = sub_img
        
        
    cv2.circle(img, img_cntr, 10, (0, 0, 255), 20)
    
    img = cv2.resize(img, [252*4, 252*3])
    #img = cv2.resize(img, [504*4, 504*3])    
               
    cv2.imshow(f'{idx} {el}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    #break


# In[50]:


img.shape


# In[48]:



sub_img = img[results.xyxy[0][target_goal][0].int().item() : results.xyxy[0][target_goal][1].int().item(), 
              results.xyxy[0][target_goal][2].int().item() : results.xyxy[0][target_goal][3].int().item()
             ]
sub_img


# In[45]:


sub_img = cv2.resize(sub_img, [252*4, 252*3])
#img = cv2.resize(img, [504*4, 504*3])    
           
cv2.imshow(f'{idx} {el}', sub_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[ ]:





# In[52]:


cv2.imshow(f'{idx} {el}', img[results.xyxy[0][target_goal][0].int().item() : results.xyxy[0][target_goal][1].int().item(), 
          results.xyxy[0][target_goal][2].int().item() : results.xyxy[0][target_goal][3].int().item()
         ])
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[53]:


results.xyxy[0][target_goal][0].int().item(), results.xyxy[0][target_goal][1].int().item(),  results.xyxy[0][target_goal][2].int().item(), results.xyxy[0][target_goal][3].int().item()


# In[54]:


'x_min', 'y_min', 'x_max', 'y_max', 'conf',


# In[ ]:





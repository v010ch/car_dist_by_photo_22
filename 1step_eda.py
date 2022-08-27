#!/usr/bin/env python
# coding: utf-8

# In[36]:


import cv2

from PIL import Image
from PIL.ExifTags import TAGS

#import pillow_heif
#from pillow_heif import register_heif_opener
#register_heif_opener() # for using Image.open for .heic without changes


# In[3]:


import os
from collections import Counter
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
tqdm.pandas()


# In[4]:


#import seaborn as sns


# In[5]:


#import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#!pip list


# In[ ]:





# In[ ]:





# In[8]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# In[35]:


def open_img(inp_path: str) -> np.ndarray:
    
    #if inp_path.endswith('.jpg'):
    ret_img = cv2.imread(inp_path)
    #else:
    #    if pillow_heif.is_supported(inp_path):
    #        heif_file = pillow_heif.open_heif(inp_path, convert_hdr_to_8bit=False)
            #print("image mode:", heif_file.mode)
            #print("image data length:", len(heif_file.data))
            #print("image data stride:", heif_file.stride)
    #        if heif_file.has_alpha:
    #            heif_file.convert_to("BGRA;16")
    #        else:
    #            heif_file.convert_to("BGR;16")  # convert 10 bit image to RGB 16 bit.
            #print("image mode:", heif_file.mode)
    #        ret_img = np.asarray(heif_file)
    
    return ret_img


# In[10]:


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

# In[11]:


train_list = os.listdir(DIR_DATA_TRAIN)
test_list  = os.listdir(DIR_DATA_TEST)

train_heic = [el for el in train_list if el.endswith('.heic')]
test_heic  = [el for el in test_list  if el.endswith('.heic')]

train_jpg = [el for el in train_list if el.endswith('.jpg')]
test_jpg  = [el for el in test_list if el.endswith('.jpg')]

fnames_train = set([el.split('.')[0] for el in train_list])
fnames_test  = set([el.split('.')[0] for el in test_list])


# In[12]:


#print(Counter([el.split('.')[1] for el in train_list]))
#print(Counter([el.split('.')[1] for el in test_list]))


# In[ ]:





# ### Загружаем данные

# (train_upd загружается только после выполнения 2step_make_features)

# In[13]:


#train_df = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), delimiter = ';')
train_df = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))
test_df  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))
train_df.shape, test_df.shape


# In[14]:


train_df.head()


# In[15]:


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

# In[16]:


train_df.distance.nlargest(5)


# In[17]:


train_df.distance.nsmallest(5)


# In[18]:


print('min ', train_df.distance[train_df.distance.argmin()], '  ', train_df.image_name[train_df.distance.argmin()])
print('max ', train_df.distance[train_df.distance.argmax()], '  ', train_df.image_name[train_df.distance.argmax()])


# In[19]:


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





# In[20]:


train_df['ext'] = train_df.image_name.apply(lambda x: x.split('.')[1])


# In[21]:


train_df.distance.hist(bins = 15)


# In[22]:


# 520 x 112
# 245 x 160
# 290 х 170


# In[ ]:





# Посмотрим пересечение датасетов

# In[23]:


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
# In[24]:


motion_blur_train = ['img_2709.heic', 'img_2733.heic', 'img_2734.heic']    # 'img_2734.heic' возможно рабочий 
motion_blur_test  = ['img_2674.heic']


# In[25]:


img = open_img(os.path.join(DIR_DATA_TRAIN, 'img_2745.heic'))
#img = open_img(os.path.join(DIR_DATA_TEST, motion_blur_test[0]))
img = cv2.resize(img, [252*4, 252*3])
               
cv2.imshow('motion blur', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[ ]:





# In[26]:


train_df.columns


# In[27]:


#train_df['exp_w'] = train_df.w.apply(lambda x: np.exp(x))
train_df['log_w'] = train_df.w.apply(lambda x: np.log(x))


# In[ ]:





# Посмотрим на корреляцию с признаками из train_upd

# In[31]:


#plot_corrc(train_df, ['x_min', 'y_min', 'x_max', 'y_max', 'h', 'w']) #'conf', 
plot_corrc(train_df, ['w'])


# In[ ]:





# In[29]:


train_df.sort_values('w').head(5)


# In[30]:


test_df.sort_values('w').head(5)


# In[30]:


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





# In[31]:


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





# In[33]:


colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), }


# In[37]:


import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0, 2]  # person and car

_ = model.cpu()


# In[38]:


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


# In[39]:


#img = open_img(os.path.join(DIR_DATA_TRAIN, 'img_2733.heic'))
img = open_img(os.path.join(DIR_DATA_TRAIN, 'img_2674.jpg'))
#results = model(np.array(img))
results = model(img)


# In[40]:


colors[results.xyxy[0][0][-1].int().item()]


# In[41]:


def get_car_center(inp_tensor: torch.Tensor) -> Tuple[int, int]:

    car_cntr = (int((inp_tensor[2].int().item() - inp_tensor[0].int().item())/2 + inp_tensor[0].int().item()),
                int((inp_tensor[3].int().item() - inp_tensor[1].int().item())/2 + inp_tensor[1].int().item())
        )
    
    return car_cntr


# In[42]:


def get_center_dist(inp_center: Tuple[int, int], inp_point: Tuple[int, int]) -> float:
    
    return np.sqrt((inp_center[0] - inp_point[0])**2 +                    (inp_center[1] - inp_point[1])**2)


# In[43]:


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


# In[44]:


def get_aspect_ratio(inp_points: np.ndarray) -> float:
    upper_len  = inp_points[1, 0, 0] - inp_points[0, 0, 0] 
    bottom_len = inp_points[2, 0, 0] - inp_points[3, 0, 0] 
    aver_len   = (upper_len + bottom_len)/2
    
    upper_hi  = inp_points[3, 0, 1] - inp_points[0, 0, 1] 
    bottom_hi = inp_points[2, 0, 1] - inp_points[1, 0, 1] 
    aver_hi   = (upper_hi + bottom_hi)/2
    
    return aver_hi / aver_len


# In[45]:


#def get_label_plate_features(inp_img: np.ndarray, inp_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
def get_label_plate_features(inp_img: np.ndarray, inp_coords: np.ndarray) -> List[float]:
    
    ret_lp_region   = np.zeros((4, 1, 2), dtype = np.int32)
    ret_legal_plate = np.zeros((4, 1, 2), dtype = np.int32)
    
    #x_min, y_min, x_max, y_max, conf, class, 
    #sub_img = inp_img[int(inp_coords.y_min) : int(inp_coords.y_max),
    #                  int(inp_coords.x_min) : int(inp_coords.x_max)
    #                 ]
    sub_img = inp_img[int(inp_coords[1]) : int(inp_coords[3]),
                      int(inp_coords[0]) : int(inp_coords[2])
                     ]
    
    # немного размываем что бы убрать лишние грани
    sub_img = cv2.bilateralFilter(sub_img, 11, 17, 17) 
    
    # детекрируем грани / контуры
    #edged = cv2.Canny(sub_img, 30, 200) 
    edged = cv2.Canny(sub_img, 15, 400) 
    cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # находим примерный контур номерного знака среди 10 самых больших контуров
    # 
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:10]   
    for idx, c in enumerate(cnts):
        # вычисляем периметр контура. True - контур замкнутый
        perimeter = cv2.arcLength(c, True)                     
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)  # 
        
        # предполагаем, что если контур из 4х точек, что это вероятно номер
        # так же учитываем соотношение сторон
        # ??? ограничить угол???
        if len(approx) == 4:  
            ar = get_aspect_ratio(approx) 
            #print(idx, ar)
            if ar < 0.85 and ar > 0.65:
                #print('region ', idx, ar)
                ret_lp_region = approx

            if ar < 0.25 and ar > 0.15:
                #print('legal plate ', idx, ar)
                ret_legal_plate = approx
                
    return (ret_legal_plate, ret_lp_region)
    
    # извлекаем признаки из координат
    #ret_legal_plate = get_lp_features_by_coord(ret_legal_plate)
    #ret_lp_region   = get_lp_features_by_coord(ret_lp_region)
    
    #return ret_legal_plate + ret_lp_region


# In[ ]:




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
# In[46]:


img_cntr = (int(img.shape[1]/2), int(img.shape[0]/2))
target_goal = determine_targ_car(results, img_cntr)


# In[47]:


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
# In[48]:


img = cv2.resize(img, [252*4, 252*3])
               
cv2.imshow('motion blur', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[53]:


train_df[train_Df.image_name == 'image_2321.jpg']


# In[ ]:





# In[57]:


tt = ['img_1927.jpg', 'img_2321.jpg', 'img_2577.jpg','img_2578.jpg','img_2579.jpg','img_2583.jpg','img_2694.jpg',]
#for idx, el in enumerate(train_list[:10]):
#for idx, el in enumerate(train_list[100:120]):
for idx, el in enumerate(tt):
    #img = open_img(os.path.join(DIR_DATA_TRAIN, el))
    img = Image.open(os.path.join(DIR_DATA_TRAIN, el))
    img = np.array(img)
    
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

        plate, reg = get_label_plate_features(img, results.xyxy[0][target_goal].numpy().tolist())
        sub_img = img[results.xyxy[0][target_goal][1].int().item() : results.xyxy[0][target_goal][3].int().item(), 
                  results.xyxy[0][target_goal][0].int().item() : results.xyxy[0][target_goal][2].int().item()
                 ]
        cv2.drawContours(sub_img, [plate], -1, (255, 0, 0), 6)        
        cv2.drawContours(sub_img, [reg], -1, (0, 0, 255), 6) 

        img[results.xyxy[0][target_goal][1].int().item() : results.xyxy[0][target_goal][3].int().item(), 
            results.xyxy[0][target_goal][0].int().item() : results.xyxy[0][target_goal][2].int().item()
          ] = sub_img
    
        
        
        
        
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
    
    #break


# In[89]:


plate


# In[50]:


print(results.xyxy[0][target_goal][1].int().item(), results.xyxy[0][target_goal][3].int().item(), )
print(results.xyxy[0][target_goal][0].int().item(), results.xyxy[0][target_goal][2].int().item())


# In[51]:


sub_img = img[results.xyxy[0][target_goal][1].int().item() : results.xyxy[0][target_goal][3].int().item(), 
              results.xyxy[0][target_goal][0].int().item() : results.xyxy[0][target_goal][2].int().item()
             ]

sub_img = cv2.resize(sub_img, [252*4, 252*3])
#img = cv2.resize(img, [504*4, 504*3])    

cv2.imshow(f'tt', sub_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[47]:


sub_img


# In[59]:


train_df = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))


# In[71]:


tmp99.image_name.values[0]


# In[72]:


tt = ['img_1927.jpg', 'img_2321.jpg', 'img_2577.jpg','img_2578.jpg','img_2579.jpg','img_2583.jpg','img_2694.jpg',]
for el in tt:
    tmp99 = train_df[train_df.image_name == el]
    
    img = Image.open(os.path.join(DIR_DATA_TRAIN, tmp99.image_name.values[0]))
    img = np.array(img)
    
    sub_img = img[int(tmp99.y_min) : int(tmp99.y_max),
                  int(tmp99.x_min) : int(tmp99.x_max)
                 ]

    sub_img = cv2.resize(sub_img, [252*4, 252*3])
    #img = cv2.resize(img, [504*4, 504*3])    

    cv2.imshow(f'tt', sub_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    


# In[67]:


tmp99.y_min


# In[ ]:





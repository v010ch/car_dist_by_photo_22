#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '')


# In[2]:


import time
notebookstart= time.time()


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


#import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





# In[8]:


#import skimage
#print(skimage.__version__)


# In[ ]:





# Блок для воспроизводимости результата

# In[9]:


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





# In[10]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_TRAIN = os.path.join(os.getcwd(), 'subm', 'train')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# # Загружаем и подготавливаем данные

# In[11]:


test_img_names  = set(os.listdir(DIR_DATA_TEST))
train_img_names = set(os.listdir(DIR_DATA_TRAIN))
len(test_img_names), len(train_img_names)


# In[12]:


train_labels_df = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), sep=';', index_col=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


def get_car_center(inp_tensor: torch.Tensor) -> Tuple[int, int]:

    car_cntr = (int((inp_tensor[2].int().item() - inp_tensor[0].int().item())/2 + inp_tensor[0].int().item()),
                int((inp_tensor[3].int().item() - inp_tensor[1].int().item())/2 + inp_tensor[1].int().item())
        )
    
    return car_cntr


# In[14]:


def get_center_dist(inp_center: Tuple[int, int], inp_point: Tuple[int, int]) -> float:
    
    return np.sqrt((inp_center[0] - inp_point[0])**2 +                    (inp_center[1] - inp_point[1])**2)


# In[15]:


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


# In[23]:


def create_car_feeatures_yolo(inp_fnames: List[str], inp_dir: str, inp_model, use_centr: Optional[bool] = False) -> pd.DataFrame:
    
    ret_data = []

    for img_name in tqdm(inp_fnames): 

        img = Image.open(os.path.join(inp_dir, img_name))
        
        
        img = np.array(img)
        #results = model(np.array(img))
        results = inp_model(img)
    
        # найден хотя бы один объект
        if results.xyxy[0].shape != torch.Size([0, 6]):

            # искать ближайший к центру кадра объект?   
            if use_centr:
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
            
            
            #get_label_plate_features(img, results)
            
            
        else:
            print(f'wtf, {img_name}   {results.xyxy[0].shape}')
            # позволим алгоритмам самим выбирать как заполнить пропуски
            #results = [0, 0, 0, 0, 0, 0, 0, 0]

# позволим алгоритмам самим выбирать как заполнить пропуски
#        ret_data.append([img_name] + results)
        
    ret_data = pd.DataFrame(ret_data, columns = ['image_name', 'car_x_min', 'car_y_min', 'car_x_max', 'car_y_max', 'car_conf', 'car_class', 'car_h', 'car_w'])
        
    return ret_data


# In[24]:


def create_license_plate_feeatures_yolo(inp_df: pd.DataFrame, inp_dir: str, inp_model, use_centr: Optional[bool] = False) -> pd.DataFrame:
    
    
    for el in inp_df.index:
        img = Image.open(os.path.join(inp_dir, img_name))
        img = np.array(img)
        #results = model(np.array(img))
        results = inp_model(img)
    
        # найден хотя бы один объект
        if results.xyxy[0].shape != torch.Size([0, 6]):
            pass   
            
            
            
    pass


# In[ ]:





# In[25]:


#model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  #
_ = model.cpu()


#model_plate = torch.hub.load('ultralytics/yolov5', 'custom', path = './models/best_y5m_full_4e.pt', source='local')
#model_plate = torch.hub.load('ultralytics/yolov5', 'custom', path = './models_weights/best_y5l_full_3e.pt', force_reload=True)
#model_plate = torch.load('./models/last_y5m_full_4e.pt')


# In[ ]:





# In[26]:


#model

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
# In[27]:


train_df = create_car_feeatures_yolo(train_img_names, DIR_DATA_TRAIN, model, use_centr = True) #use_centr
#train_df = create_car_feeatures_resnet(train_img_names, DIR_DATA_TRAIN, model, use_centr = True) #use_centr
train_df = pd.merge(train_labels_df, train_df, how='left')
train_df.shape


# In[ ]:





# In[29]:


test_df = create_car_feeatures_yolo(test_img_names, DIR_DATA_TEST, model, use_centr = True) #use_centr
test_df.shape


# In[ ]:





# yolov5 не найдено машин:     
# train:   
# img_1890.jpg (w&h < 200),     
# 
# test: 
# img_1888.jpg (w&h < 200), 
# img_1889.jpg(only person)
# img_2674.heic, 
# img_2571.jpg (w&h < 200), 

# In[ ]:





# In[31]:


sns.histplot(train_df, x='car_h')
plt.show()


# In[32]:


sns.histplot(train_df, x='car_w')
plt.show()


# In[ ]:





# In[34]:


train_df['car_class'].value_counts()


# In[35]:


test_df['car_class'].value_counts()

2.0    525
0.0      42.0    514
0.0      4
# In[ ]:





# In[38]:


for el in ['car_x_min', 'car_y_min', 'car_x_max', 'car_y_max', 'car_h', 'car_w']:
    train_df[f'log_{el}'] = train_df[el].apply(lambda x: np.log(x))
    test_df[f'log_{el}'] = test_df[el].apply(lambda x: np.log(x))


# In[ ]:





# In[40]:


train_df.head(10)


# In[ ]:





# In[41]:


train_df.to_csv(os.path.join(DIR_DATA, 'train_upd.csv'), index = False)
test_df.to_csv(os.path.join(DIR_DATA, 'test_upd.csv'), index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




import cv2train_df = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))
test_df = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))def get_aspect_ratio(inp_points: np.ndarray) -> float:
    upper_len  = inp_points[1, 0, 0] - inp_points[0, 0, 0] 
    bottom_len = inp_points[2, 0, 0] - inp_points[3, 0, 0] 
    aver_len   = (upper_len + bottom_len)/2
    
    upper_hi  = inp_points[3, 0, 1] - inp_points[0, 0, 1] 
    bottom_hi = inp_points[2, 0, 1] - inp_points[1, 0, 1] 
    aver_hi   = (upper_hi + bottom_hi)/2
    
    return aver_hi / aver_lenafter cv2.approxPolyDP
ndarray[idx][0][x / y]
0------>x
|
|
|
v
y

0 - [left_uper] # x y
1 - [right_upper] 
2 - [right_botttom]
3 - [left_bottom]def get_lp_features_by_coord(inp_coords: np.ndarray) -> List[float]:
    
    h_min = inp_coords[0, 0, 1] - inp_coords[3, 0, 1]
    h_max = inp_coords[1, 0, 1] - inp_coords[2, 0, 1]

    w_min = inp_coords[0, 0, 0] - inp_coords[1, 0, 0]
    w_max = inp_coords[3, 0, 0] - inp_coords[2, 0, 0]

    h_aver = (h_min + h_max)/2
    w_aver = (w_min + w_max)/2

    if h_min > h_max:
        h_min, h_max = h_max, h_min

    if w_min > w_max:
        w_min, w_max = w_max, w_min

    #print(h_min, h_max, h_aver, w_min, w_max, w_aver)
    
    return [h_min, h_max, h_aver, w_min, w_max, w_aver]#def get_label_plate_features(inp_img: np.ndarray, inp_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    ret_legal_plate = get_lp_features_by_coord(ret_legal_plate)
    ret_lp_region   = get_lp_features_by_coord(ret_lp_region)
    
    return ret_legal_plate + ret_lp_regiontmp = train_df.loc[115, :]
tmpimg_cntr = True

img = Image.open(os.path.join(DIR_DATA_TRAIN, tmp.image_name))
img = np.array(img)
#results = model(np.array(img))
results = model(img)

# найден хотя бы один объект
if results.xyxy[0].shape != torch.Size([0, 6]):

    # искать ближайший к центру кадра объект?   
    if img_cntr:
        img_cntr = (int(img.shape[1]/2), int(img.shape[0]/2))
        target_goal = determine_targ_car(results, img_cntr)
    else:
        target_goal = 0

    if target_goal < 0:
        print(f'wtf2, {img_name}   {results.xyxy[0].shape}')
    else:
        h = results.xyxy[0][target_goal][3] - results.xyxy[0][target_goal][1]
        w = results.xyxy[0][target_goal][2] - results.xyxy[0][target_goal][0]
        results = results.xyxy[0][target_goal].numpy().tolist() + [h.item(), w.item()]
        
        legal_plate, lp_region = get_label_plate_features(img, results)
        #print()
        
cv2.drawContours(sub_img, [legal_plate], -1, (0, 255, 0), 2)        
cv2.drawContours(sub_img, [lp_region], -1, (0, 0, 255), 2)        
#img = cv2.resize(img, [252*4, 252*3])
sub_img = cv2.resize(sub_img, [252*4, 252*3])

#cv2.imshow('license plate', img)
cv2.imshow('license plate', sub_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '')


# In[ ]:


import time
notebookstart = time.time()


# In[ ]:


import torch


# In[ ]:


import os
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener() # for using Image.open for .heic without changes

from tqdm.auto import tqdm
tqdm.pandas()


# In[ ]:


#import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[ ]:





# # Блок для воспроизводимости результата

# In[ ]:


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





# # Выставление констант

# In[ ]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_TRAIN = os.path.join(os.getcwd(), 'subm', 'train')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# # Загрузка и подготовка данных

# In[ ]:


test_img_names  = set(os.listdir(DIR_DATA_TEST))
train_img_names = set(os.listdir(DIR_DATA_TRAIN))
len(test_img_names), len(train_img_names)


# In[ ]:


train_labels_df = pd.read_csv(os.path.join(DIR_DATA, 'train.csv'), sep=';', index_col=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def get_car_center(inp_tensor: torch.Tensor) -> Tuple[int, int]:
    """
    Получение цетра рамки автомашины в координатах всего изображения
    args:
        inp_tensor - координаты рамки автомобиля
    return:
        координаты центра
    """
    car_cntr = (int((inp_tensor[2].int().item() - inp_tensor[0].int().item())/2 + inp_tensor[0].int().item()),
                int((inp_tensor[3].int().item() - inp_tensor[1].int().item())/2 + inp_tensor[1].int().item())
        )
    
    return car_cntr


# In[ ]:


def get_center_dist(inp_center: Tuple[int, int], inp_point: Tuple[int, int]) -> float:
    """
    Получение расстояния (евклидова) от центра изображения до заданной точки.
    Заданная точка в текущей задаче - центр рамки автомобиля
    args:
        inp_center - координаты центра изображения
        inp_point  - координаты точки, до которой определяется расстояние (центр рамки автомобиля)
    return:
        float - расстояние
    """
    return np.sqrt((inp_center[0] - inp_point[0])**2 +                    (inp_center[1] - inp_point[1])**2)


# In[ ]:


def determine_targ_car(inp_results, inp_img_cntr: Tuple[int, int]) -> int:
    """
    Определение рамки целевого автомобиля:
     - класс - автомобиль
     - габариты по длине и ширине не менее 200 пикселей
     - наименьшее расстояние до центра изображения
    args:
        inp_results - координаты найденных модельб рамок
    return:
       int - порядковый номер рамки целевой автомашины
    """
    min_dist = 1000000
    min_idx  = -1
    
    for el in range(inp_results.xyxy[0].shape[0]):
        # учитываем только машины (рудимент - из модели не убрал рамки пешеходов)
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


# In[ ]:


def create_car_feeatures_yolo(inp_fnames: List[str], inp_dir: str, inp_model, use_centr: Optional[bool] = False) -> pd.DataFrame:
    """
    Создание признаков координат рамок целового автомобиля и их линейных размеров
    args:
        inp_fnames - список изображений для построения признаков
        inp_dir - директория файлов изображений
        inp_model - модель для определения рамок автомобилей на изображении
        use_centr - признак выбирать ли центральный автомобиль или 0й
    return:
        DataFrame - для каждого изображения сопоставлена строка из значений
                    car_x_min, car_y_min, car_x_max, car_y_max, car_conf, car_class, car_h, car_w
    """
    ret_data = []

    for img_name in tqdm(inp_fnames): 
        img = Image.open(os.path.join(inp_dir, img_name))
        
        
        img = np.array(img)
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
            
        # на изобрадении не найдено ни одного автомобиля, удовлетворяющего условиям.
        # просто сообщаем об этом. в датасете будут Nan для таких изображений.
        # посзволим catboost самой решить что делать с такими пропусками.
        else:
            print(f'wtf, {img_name}   {results.xyxy[0].shape}')

        
    ret_data = pd.DataFrame(ret_data, columns = ['image_name', 'car_x_min', 'car_y_min', 'car_x_max', 'car_y_max', 'car_conf', 'car_class', 'car_h', 'car_w'])
        
    return ret_data


# In[ ]:





# # Создаем признаки

# In[ ]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  #
#model_plate = torch.hub.load('ultralytics/yolov5', 'custom', path = './models/best_y5m_full_4e.pt', source='local')
#model_plate = torch.load('./models/last_y5m_full_4e.pt')

_ = model.cpu()


# In[ ]:





# In[ ]:


train_df = create_car_feeatures_yolo(train_img_names, DIR_DATA_TRAIN, model, use_centr = True) #use_centr
train_df = pd.merge(train_labels_df, train_df, how='left')
train_df.shape


# In[ ]:


test_df = create_car_feeatures_yolo(test_img_names, DIR_DATA_TEST, model, use_centr = True) #use_centr
test_df.shape


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





# In[ ]:


sns.histplot(train_df, x = 'car_h')
plt.show()


# In[ ]:


sns.histplot(test_df, x = 'car_w')
plt.show()


# In[ ]:


sns.histplot(train_df, x = 'car_w')
plt.show()


# In[ ]:


sns.histplot(test_df, x = 'car_w')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


for el in ['car_x_min', 'car_y_min', 'car_x_max', 'car_y_max', 'car_h', 'car_w']:
    train_df[f'log_{el}'] = train_df[el].apply(lambda x: np.log(x))
    test_df[f'log_{el}'] = test_df[el].apply(lambda x: np.log(x))


# In[ ]:


train_df.head(10)


# In[ ]:





# In[ ]:


train_df.to_csv(os.path.join(DIR_DATA, 'train_upd.csv'), index = False)
test_df.to_csv(os.path.join(DIR_DATA, 'test_upd.csv'), index = False)


# In[ ]:





# In[ ]:





# In[ ]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:





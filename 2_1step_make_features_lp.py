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

from torchvision import models
from torchvision import transforms 


# In[ ]:


import os
from typing import List, Tuple, Optional
from ast import literal_eval

import pandas as pd
import numpy as np

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener() # for using Image.open for .heic without changes

from tqdm.auto import tqdm
tqdm.pandas()


# In[ ]:


import cv2


# In[ ]:


get_ipython().run_line_magic('watermark', '--iversions')


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





# In[ ]:





# # Загрузка данных

# In[ ]:


train_df = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))
test_df = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))


# In[ ]:


#115 img_1824.jpg - белая машина с белой рамкой


# In[ ]:





# In[ ]:


def create_model(outputchannels: Optional[int] = 1, aux_loss: Optional[bool] = False, freeze_backbone: Optional[bool] = False):
    """
    Создание и настройка объекта модели для дальнейшей загрузки предобученных весов
    args:
        outputchannels - количество каналов для выхода модели (опционально, 1 - бинарный выход)
        aux_loss - не используется в финальном решении
        freeze_backbone - рассчитывать ли в дальнейшем обратный градиент
    return:
        настроенный объект модели
    """
    model = models.segmentation.deeplabv3_resnet101(
        pretrained = True, progress = True)#, aux_loss=aux_loss)

    if freeze_backbone is True:
        for p in model.parameters():
            p.requires_grad = False

    #model.classifier = models.segmentation.segmentation.DeepLabHead(
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(
        2048, outputchannels)

    return model


# In[ ]:


# Prediction pipeline
def pred(inp_image: np.ndarray, inp_model):
    """
    Предсказание модели. Предсказывает какие из пикселей изображения относятся к автомобильному номеру.
    args:
        inp_image - входное изображение
        inp_model - используемая модель
    return:
        облако точек, относящихся к автомобильному номеру на данном изображении
    """
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

    input_tensor = preprocess(inp_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = inp_model(input_batch)['out'][0]
    
    return output
 


# In[ ]:


def get_plate_features_tuple(inp_row: str, inp_folder: str, inp_model) -> Tuple[int, int, int, int]:
    """
    
    args:
        inp_row - строка датафрейма. из нее берутся имя файла изображения и координаты рамки целевого автомобиля
        inp_folder - папка изображений
        inp_model - используемая модель
    return:
        мин и макс x и y координаты облака точек автомобильного номера на изображении
    """
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0
    
    # найдена licence plate
    if inp_row.car_y_min > 0:

        img = Image.open(os.path.join(inp_folder, inp_row.image_name))
        img = np.array(img)
        sub_img = img[int(inp_row.car_y_min) : int(inp_row.car_y_max),
                      int(inp_row.car_x_min) : int(inp_row.car_x_max)
                     ]

        # Defining a threshold for predictions
        threshold = 0.1 # 0.1 seems appropriate for the pre-trained model

        # Predict
        output = pred(sub_img, inp_model)


        output = (output > threshold).type(torch.IntTensor)
        output_np = output.cpu().numpy()[0]

        # Extracting coordinates
        result = np.where(output_np > 0)
        coords = list(zip(result[0], result[1]))

        # интересуцют только мин и макс x и y
        if len(coords) != 0:
            x_min = sorted(coords, key = lambda x: x[0])[0][0]
            y_min = sorted(coords, key = lambda x: x[1])[0][1]
            x_max = sorted(coords, key = lambda x: x[0])[-1][0]
            y_max = sorted(coords, key = lambda x: x[1])[-1][1]
    
    return (x_min, y_min, x_max, y_max)


# In[ ]:


def get_plate_features(inp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекаем из Tuple мин и макс x и y координаты и делаем их отдельными признаками.
    Расчет длинны и ширины по x и y.
    args:
        inp_df - входящий DataFrame для преобразования
    return:
        преобразованный DataFrame
    """
    #inp_df.tmp = inp_df.tmp.apply(lambda x: (x))
    
    inp_df['plate_x_min'] = inp_df.tmp.apply(lambda x: float(x[0]))
    inp_df['plate_y_min'] = inp_df.tmp.apply(lambda x: float(x[1]))
    inp_df['plate_x_max'] = inp_df.tmp.apply(lambda x: float(x[2]))
    inp_df['plate_y_max'] = inp_df.tmp.apply(lambda x: float(x[3]))
    
    inp_df['plate_w'] = inp_df.plate_x_max - inp_df.plate_x_min
    inp_df['plate_h'] = inp_df.plate_y_max - inp_df.plate_y_min
    
    #inp_df.drop(['tmp'], axis = 0, inplace = True)
    
    return inp_df


# In[ ]:





# # Построение признаков

# In[ ]:


# Load the model:
model = create_model()
checkpoint = torch.load('./models_weights/model_v2.pth', map_location = 'cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()
_ = model.to('cpu')


# In[ ]:


print('before ', train_df.shape, test_df.shape)
train_df['tmp'] = train_df.progress_apply(lambda x: get_plate_features_tuple(x, DIR_DATA_TRAIN, model), axis = 1)
test_df['tmp']  = test_df.progress_apply(lambda x: get_plate_features_tuple(x, DIR_DATA_TEST, model), axis = 1)
print('after  ', train_df.shape, test_df.shape)


# In[ ]:


print('before ', train_df.shape, test_df.shape)
train_df = get_plate_features(train_df)
test_df  = get_plate_features(test_df)
print('after  ', train_df.shape, test_df.shape)


# In[ ]:


for el in ['plate_w', 'plate_h']:
    train_df[f'log_{el}'] = train_df[el].apply(lambda x: np.log(x))
    test_df[f'log_{el}']  = test_df[el].apply(lambda x: np.log(x))


# In[ ]:


train_df.to_csv(os.path.join(DIR_DATA, 'train_upd.csv'), index = False)
test_df.to_csv(os.path.join(DIR_DATA,  'test_upd.csv'), index = False)


# In[ ]:





# In[ ]:


test_df.head()


# In[ ]:





# In[ ]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


# In[ ]:





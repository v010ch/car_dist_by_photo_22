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

from torchvision import models
from torchvision import transforms #import (Compose, Normalize, Resize, ToPILImage,
                                   # ToTensor)


# In[4]:


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


# In[5]:


import cv2


# In[6]:


get_ipython().run_line_magic('watermark', '--iversions')


# Блок для воспроизводимости результата

# In[7]:


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





# In[8]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_TRAIN = os.path.join(os.getcwd(), 'subm', 'train')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# In[ ]:





# # Загрузка данных

# In[9]:


train_df = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))
test_df = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))


# In[10]:


#115 img_1824.jpg - белая машина с белой рамкой


# In[ ]:





# In[13]:


def create_model(outputchannels: Optional[int] = 1, aux_loss: Optional[bool] = False, freeze_backbone: Optional[bool] = False):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True)#, aux_loss=aux_loss)

    if freeze_backbone is True:
        for p in model.parameters():
            p.requires_grad = False

    #model.classifier = models.segmentation.segmentation.DeepLabHead(
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(
        2048, outputchannels)

    return model


# In[24]:


# Prediction pipeline
def pred(inp_image: np.ndarray, inp_model):
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

    input_tensor = preprocess(inp_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = inp_model(input_batch)['out'][0]
    
    return output
 

%%time
#filename = os.path.join(DIR_DATA_TRAIN, tmp.image_name)
# Loading an image
#img = Image.open(f'{filename}').convert('RGB')



tmp = train_df.loc[115, :]

img = Image.open(os.path.join(DIR_DATA_TRAIN, tmp.image_name))
img = np.array(img)
sub_img = img[int(tmp.y_min) : int(tmp.y_max),
              int(tmp.x_min) : int(tmp.x_max)
             ]

# Defining a threshold for predictions
threshold = 0.1 # 0.1 seems appropriate for the pre-trained model

# Predict
#output = pred(img, model)
output = pred(sub_img, model)

output = (output > threshold).type(torch.IntTensor)
output_np = output.cpu().numpy()[0]

# Extracting coordinates
result = np.where(output_np > 0)
coords = list(zip(result[0], result[1]))


x_min = sorted(coords, key = lambda x: x[0])[0][0]
y_min = sorted(coords, key = lambda x: x[1])[0][1]
x_max = sorted(coords, key = lambda x: x[0])[-1][0]
y_max = sorted(coords, key = lambda x: x[1])[-1][1]

# Overlay the original image
#for cord in coords:
    #frame.putpixel((cord[1], cord[0]), (255, 0, 0))
    #img.putpixel((cord[1], cord[0]), (255, 0, 0))
# In[20]:


def get_plate_features_tuple(inp_row: str, inp_folder: str, inp_model) -> Tuple[int, int, int, int]:
    
    #print(inp_row)
    #return 0

    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0
    
    # найдена licence plate
    if inp_row.car_y_min > 0:

        #img = Image.open(os.path.join(DIR_DATA_TRAIN, tmp.image_name))
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

        if len(coords) != 0:
            x_min = sorted(coords, key = lambda x: x[0])[0][0]
            y_min = sorted(coords, key = lambda x: x[1])[0][1]
            x_max = sorted(coords, key = lambda x: x[0])[-1][0]
            y_max = sorted(coords, key = lambda x: x[1])[-1][1]
    
    return (x_min, y_min, x_max, y_max)


# In[21]:


def get_plate_features(inp_df: pd.DataFrame) -> pd.DataFrame:
    
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





# In[22]:


#https://github.com/dennisbappert/pytorch-licenseplate-segmentation
# Load the model:
model = create_model()
checkpoint = torch.load('./models_weights/model_v2.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()
_ = model.to('cpu')


# In[25]:


print('before ', train_df.shape, test_df.shape)
train_df['tmp'] = train_df.progress_apply(lambda x: get_plate_features_tuple(x, DIR_DATA_TRAIN, model), axis = 1)
test_df['tmp'] = test_df.progress_apply(lambda x: get_plate_features_tuple(x, DIR_DATA_TEST, model), axis = 1)
print('after  ', train_df.shape, test_df.shape)


# In[26]:


print('before ', train_df.shape, test_df.shape)
train_df = get_plate_features(train_df)
test_df = get_plate_features(test_df)
print('after  ', train_df.shape, test_df.shape)


# In[29]:


for el in ['plate_w', 'plate_h']:
    train_df[f'log_{el}'] = train_df[el].apply(lambda x: np.log(x))
    test_df[f'log_{el}'] = test_df[el].apply(lambda x: np.log(x))


# In[30]:


train_df.to_csv(os.path.join(DIR_DATA, 'train_upd.csv'), index = False)
test_df.to_csv(os.path.join(DIR_DATA,  'test_upd.csv'), index = False)


# In[ ]:





# In[28]:


test_df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




def get_aspect_ratio(inp_points: np.ndarray) -> float:
    
    upper_len  = inp_points[1, 0, 0] - inp_points[0, 0, 0] 
    bottom_len = inp_points[2, 0, 0] - inp_points[3, 0, 0] 
    aver_len   = (upper_len + bottom_len)/2
    
    upper_hi  = inp_points[3, 0, 1] - inp_points[0, 0, 1] 
    bottom_hi = inp_points[2, 0, 1] - inp_points[1, 0, 1] 
    aver_hi   = (upper_hi + bottom_hi)/2
    
    return aver_hi / aver_len# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def apply_brightness_contrast(input_img: np.ndarray, brightness: Optional[int] = 0, contrast: Optional[int] = 0) -> np.ndarray:
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buftmp = train_df.loc[115, :]

img = Image.open(os.path.join(DIR_DATA_TRAIN, tmp.image_name))
img = np.array(img)

sub_img = img[int(tmp.y_min) : int(tmp.y_max),
              int(tmp.x_min) : int(tmp.x_max)
             ]
ttl_img = sub_img.copy()


# ------------ контраст
contrasted_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(contrasted_img)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l_channel)
# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl,a,b))

# Converting image from LAB Color model to BGR color spcae
contrasted_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


#alpha = 1.5 # Contrast control (1.0-3.0)
#beta = 0 # Brightness control (0-100)
#contrasted_img = cv2.convertScaleAbs(sub_img, alpha=alpha, beta=beta)


ttl_img = np.concatenate([ttl_img, contrasted_img], axis = 1)


# ------------ грани
#gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) 
#gray_image = cv2.bilateralFilter(sub_img, 11, 17, 17) 
gray_image = cv2.bilateralFilter(contrasted_img, 11, 17, 17) 

edged = cv2.Canny(gray_image, 100, 200) 
#edged = cv2.Canny(gray_image, 5, 200) 
edged2 = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
ttl_img = np.concatenate([ttl_img, edged2], axis = 1)



# ------------ контуры
cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
sub_img1 = sub_img.copy()
cv2.drawContours(sub_img1, cnts, -1, (0, 255, 0), 3)
ttl_img = np.concatenate([ttl_img, sub_img1], axis = 1)
#ttl_img2 = sub_img1.copy()


# ------------ 10 крупнейших контуров
cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:10]
sub_img2 = sub_img.copy()
cv2.drawContours(sub_img2, cnts, -1, (0, 255, 0) ,3)
#ttl_img2 = np.concatenate([ttl_img2, sub_img2], axis = 1)
ttl_img2 = sub_img2.copy()


# ------------ вычленение номера и региона среди контуров
lp_region   = np.zeros((4, 1, 2), dtype = np.int32)
legal_plate = np.zeros((4, 1, 2), dtype = np.int32)
sub_img4 = sub_img.copy()
approx_list = []
for idx, c in enumerate(cnts):
    
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.005 * perimeter, True) #0.018
    approx_list.append(approx)
    
    cv2.drawContours(sub_img4, [approx], -1, (0, 255, 0) ,3)
    #if len(approx) == 4:
    #    ar = get_aspect_ratio(approx) 
    #    if ar < 0.8 and ar > 0.65:
    #        print('region ', idx, ar)
    #        lp_region = approx
    #        
    #    if ar < 0.25 and ar > 0.15:
    #        print('legal plate ', idx, ar)
    #        legal_plate = approx


sub_img5 = sub_img.copy()
for idx, el in enumerate(approx_list):
    
    perimeter = cv2.arcLength(el, True)
    approx = cv2.approxPolyDP(el, 0.022 * perimeter, True) #0.018
    
    cv2.drawContours(sub_img5, [approx], -1, (0, 255, 0) ,3)
    print(idx, len(approx))
    if len(approx) == 4:
        ar = get_aspect_ratio(approx) 
        print(idx, len(approx), ar)
        if ar < 0.86 and ar > 0.65:
            print('region ', idx, ar)
            lp_region = approx
            
        if ar < 0.26 and ar > 0.15:
            print('legal plate ', idx, ar)
            legal_plate = approx

            

            
            
ttl_img2 = np.concatenate([ttl_img2, sub_img4], axis = 1)
ttl_img2 = np.concatenate([ttl_img2, sub_img5], axis = 1)
            
sub_img3 = sub_img.copy()
cv2.drawContours(sub_img3, [legal_plate], -1, (0, 255, 0) ,3)
if lp_region[0, 0, 0] != 0:
    cv2.drawContours(sub_img3, [lp_region], -1, (0, 0, 255), 3)
ttl_img2 = np.concatenate([ttl_img2, sub_img3], axis = 1)



ttl_img = np.concatenate([ttl_img, ttl_img2])

ttl_img = cv2.resize(ttl_img, [1780, 814])

cv2.imshow("Top 10 contours", ttl_img)
cv2.waitKey(0)
cv2.destroyAllWindows()tmp = train_df.loc[115, :]

img = Image.open(os.path.join(DIR_DATA_TRAIN, tmp.image_name))
img = np.array(img)

sub_img = img[int(tmp.y_min) : int(tmp.y_max),
              int(tmp.x_min) : int(tmp.x_max)
             ]

# add contrast
#contrasted_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2LAB)
#l_channel, a, b = cv2.split(contrasted_img)

#clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4,4))
#cl = clahe.apply(l_channel)
# merge the CLAHE enhanced L-channel with the a and b channel
#limg = cv2.merge((cl,a,b))


# Converting image from LAB Color model to BGR color spcae
#contrasted_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)



alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

contrasted_img = cv2.convertScaleAbs(sub_img, alpha=alpha, beta=beta)



ttl_img = np.concatenate([sub_img, contrasted_img], axis = 1)

cv2.imshow("original image", ttl_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# In[ ]:





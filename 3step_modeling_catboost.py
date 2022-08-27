#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '')


# In[2]:


import time
notebookstart= time.time()


# In[3]:


import os
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor
from catboost import Pool, cv


# In[4]:


get_ipython().run_line_magic('watermark', '--iversions')


# In[5]:


from catboost import __version__ as cb_version
print(f'cb_version: {cb_version}')


# Блок для воспроизводимости результатов

# In[6]:


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





# In[7]:


DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_SUBM = os.path.join(os.getcwd(), 'subm')
DIR_SUBM_TRAIN = os.path.join(os.getcwd(), 'subm', 'train')
DIR_DATA_TRAIN = os.path.join(DIR_DATA, 'train')
DIR_DATA_TEST  = os.path.join(DIR_DATA, 'test')


# In[ ]:





# In[8]:


def plot_feature_importance(importance, names: List[str], model_type:str, imp_number: Optional[int] = 30) -> None:
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'][:imp_number], y=fi_df['feature_names'][:imp_number])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# In[ ]:





# In[ ]:





# In[ ]:





# # Загрузка данных

# In[ ]:


train_df = pd.read_csv(os.path.join(DIR_DATA, 'train_upd.csv'))
test_df  = pd.read_csv(os.path.join(DIR_DATA, 'test_upd.csv'))
train_df.shape, test_df.shape


# In[ ]:


train_df.head()


# In[ ]:





# Исключаем из обучающей выборки сильно размытые кадры из подземки.   
# На них размеры определяются явно некорректно,что помешает при обучении

# In[ ]:


motion_blur_train = set(['img_2709.heic', 'img_2733.heic', 'img_2734.heic'])    # 'img_2734.heic' возможно рабочий 
print(train_df.shape)

for el in train_df.index:
    if train_df.loc[el, 'image_name'] in motion_blur_train:
        train_df.drop(el, inplace = True)

print(train_df.shape)


# In[ ]:





# In[ ]:





# # Обучаем модель

# cv на 3 фолда.   
# по ним выбираем лучшее количество итераций по RMSE.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'params = {"iterations": 3500,\n          "loss_function": \'RMSE\',\n          #\'eval_metric\': \'R2\',\n         }\n\n#features = [\'log_x_min\', \'log_y_min\', \'log_x_max\', \'log_y_max\', \'conf\', \'log_h\', \'log_w\']\nfeatures = [\'log_plate_h\', \'log_plate_w\']\n\ntrain = Pool(data = train_df[features],\n             label = train_df[[\'distance\']],\n            )\n\nscores = cv(train, params,\n            fold_count = 3,\n            verbose = False,\n            plot = True,\n           )')


# выбираем оптимальное количество итераций

# In[ ]:


niter = scores['test-RMSE-mean'].argmin() + 13
scores['test-RMSE-mean'].min(), scores['test-RMSE-mean'].argmin(), niter

0.9254814620106862, 444    x/y/conf
0.8717672874064516, 445    x/y/conf/h/w with 0
0.8689944606627914, 608,   x/y/conf/h/w with nulls
0.8423961974720088, 271    x/y/conf/h/w with nulls   yolov5x6
0.6345816635973699, 221    x/y/conf/h/w with nulls   yolov5l cntr
0.5937488485807697, 300    x/y/conf/h/w with nulls   yolov5l cntr niter
0.570131014775917,  295    x/y/conf/h/w with nulls   yolov5l cntr niter minhw200 log

linux + env
0.5647743814930765, 705    x/y/conf/h/w with nulls   yolov5l cntr niter minhw200
# In[ ]:





# Обучем на этом количестве итераций модель на всей обучающей выборке

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nmodel_cb = CatBoostRegressor(iterations = niter, verbose = 100)\n# Fit model\nmodel_cb.fit(train_df[features], train_df[['distance']].values)")

999:	learn: 0.2699253	total: 1.49s	remaining: 0us       x/y/conf
999:	learn: 0.2365099	total: 1.23s	remaining: 0us       x/y/conf/h/w with nulls
999:	learn: 0.2223038	total: 1.57s	remaining: 0us       x/y/conf/h/w with nulls yolov5x6
999:	learn: 0.1702023	total: 1.19s	remaining: 0us       x/y/conf/h/w with nulls yolo5l cntr
312:	learn: 0.1999264	total: 440ms	remaining: 0us       x/y/conf/h/w with nulls yolo5l cntr niter
307:	learn: 0.1878093	total: 352ms	remaining: 0us       x/y/conf/h/w with nulls yolov5l cntr niter minhw200 log

linux + env
717:	learn: 0.1719744	total: 439ms	remaining: 0us       x/y/conf/h/w with nulls   yolov5l cntr niter minhw200
# Посмотрим на выжность признаков

# In[ ]:


#plot_feature_importance(model_cb.get_feature_importance(), train_df[features].get_feature_names(), 'CATBOOST')
plot_feature_importance(model_cb.get_feature_importance(), features, 'CATBOOST')


# In[ ]:





# Предскажем на трейне для просмотра ошибок в 5step

# In[ ]:


pred_train = model_cb.predict(train_df[features])
train_df['pred'] = pred_train
train_df.to_csv(os.path.join(DIR_SUBM_TRAIN, 'train_with_pred.csv'), index = False)


# In[ ]:





# In[ ]:





# Предсказываем для теста.    
# Оформляем сабмит.

# In[ ]:


preds = model_cb.predict(test_df[features])
test_df['distance'] = preds

sample_solution_df = test_df[['image_name', 'distance']]


# In[ ]:


test_img_names = set(os.listdir(DIR_DATA_TEST))


# In[ ]:


lost_test_items = []

for file_name in test_img_names - set(sample_solution_df['image_name'].values):
    lost_test_items.append([file_name, 0])
    
lost_test_items_df = pd.DataFrame(lost_test_items, columns=['image_name', 'distance'])
sample_solution_df = pd.concat([sample_solution_df, lost_test_items_df])

#sample_solution_df.to_csv(os.path.join(DIR_SUBM, '17_palte_minhw_cntr_niter_nomb.csv'), sep=';', index=False)
sample_solution_df.to_csv(os.path.join(DIR_SUBM, '18_repeat.csv'), sep=';', index=False)


# In[ ]:





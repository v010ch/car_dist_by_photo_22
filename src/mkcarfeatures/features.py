#!/usr/bin/env python
# coding: utf-8

import torch
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener() # for using Image.open for .heic without changes


import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from tqdm.auto import tqdm



def get_car_center(inp_tensor: torch.Tensor) -> Tuple[int, int]:

    car_cntr = (int((inp_tensor[2].int().item() - inp_tensor[0].int().item())/2 + inp_tensor[0].int().item()),
                int((inp_tensor[3].int().item() - inp_tensor[1].int().item())/2 + inp_tensor[1].int().item())
        )
    
    return car_cntr



def get_center_dist(inp_center: Tuple[int, int], inp_point: Tuple[int, int]) -> float:
    
    return np.sqrt((inp_center[0] - inp_point[0])**2 + (inp_center[1] - inp_point[1])**2)




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



def create_feeatures(inp_fnames: List[str], inp_dir: str, inp_model, use_centr: Optional[bool] = False):
    
    ret_data = []

    for img_name in tqdm(inp_fnames): 
        #if 'heic' in img_name:
        #    heif_file = pyheif.read(os.path.join(inp_dir, img_name))
        #   img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride)
        #else:
        #    img = Image.open(os.path.join(inp_dir, img_name))
        img = Image.open(os.path.join(inp_dir, img_name))
        
        
        img = np.array(img)
        #results = model(np.array(img))
        results = inp_model(img)
    
        if results.xyxy[0].shape != torch.Size([0, 6]):

            if use_centr:
                #img_cntr = (int(img_.shape[1]/2), int(img_.shape[0]/2))
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
            
            
        else:
            print(f'wtf, {img_name}   {results.xyxy[0].shape}')
            # позволим алгоритмам самим выбирать как заполнить пропуски
            #results = [0, 0, 0, 0, 0, 0, 0, 0]

# позволим алгоритмам самим выбирать как заполнить пропуски
#        ret_data.append([img_name] + results)
        
    ret_data = pd.DataFrame(ret_data, columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class', 'h', 'w'])
        
    return ret_data



def create_feeatures_mp(inp_img_name: str, inp_dir: str, inp_model, use_centr: Optional[bool] = False):
#def create_feeatures_mp(inp_fnames: List[str], inp_dir: str, use_centr: Optional[bool] = False):
    
    img = Image.open(os.path.join(inp_dir, inp_img_name))
    img = np.array(img)
        
    #inp_,odel = model
    results = inp_model(img)

    if results.xyxy[0].shape != torch.Size([0, 6]):

        if use_centr:
            #img_cntr = (int(img_.shape[1]/2), int(img_.shape[0]/2))
            img_cntr = (int(img.shape[1]/2), int(img.shape[0]/2))
            target_goal = determine_targ_car(results, img_cntr)
        else:
            target_goal = 0

        if target_goal < 0:
            print(f'wtf2, {inp_img_name}   {results.xyxy[0].shape}')
            results = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            #ret_data = ['fault'] + results
            ret_data = [inp_img_name] + results

        h = results.xyxy[0][target_goal][3] - results.xyxy[0][target_goal][1]
        w = results.xyxy[0][target_goal][2] - results.xyxy[0][target_goal][0]
        results = results.xyxy[0][target_goal].numpy().tolist() + [h.item(), w.item()]

        # позволим алгоритмам самим выбирать как заполнить пропуски
        ret_data = [inp_img_name] + results


    else:
        print(f'wtf, {inp_img_name}   {results.xyxy[0].shape}')
        # позволим алгоритмам самим выбирать как заполнить пропуски
        #results = [0, 0, 0, 0, 0, 0, 0, 0]
        results = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        #ret_data = ['fault'] + results
        ret_data = [inp_img_name] + results

    return ret_data

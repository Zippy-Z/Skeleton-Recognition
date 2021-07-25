# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:49:36 2021

@author: huangpj
"""

import numpy as np
from tool.visualise import visualise
from tool.graph import Graph
import glob
import os

#镜像增强
for i in range(5):
    path='./data/train_enhance/00'+str(i)
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path+"创建成功")
    else:
        print(path+"目录已存在")
for i in range(5):
    path='./data/train/00'+str(i)+'/*.npy'
    for ind, sample_path in enumerate(glob.glob(path)):
        sample = np.load(sample_path)
        outfile = './data/train_enhance/00' + str(i) + '/'+str(ind)+'.npy'
        np.save(outfile, sample)
for i in range(5):
    path='./data/train/00'+str(i)+'/*.npy'
    for ind, sample_path in enumerate(glob.glob(path)):
        sample = np.load(sample_path)
        temp = sample
        for j in range(8):
            temp[:, 0, :, j, :] = sample[:, 0, :, 17 - 1 - j, :]
            temp[:, 0, :, 17 - 1 - j, :] = sample[:, 0, :, j, :]
        outfile = './data/train_enhance/00' + str(i) + '/'+str(ind)+'_morror.npy'
        np.save(outfile, temp)

#胖瘦变换
for i in range(5):
    path='./data/train/00'+str(i)+'/*.npy'
    for ind, sample_path in enumerate(glob.glob(path)):
        sample = np.load(sample_path)
        temp = sample
        for j in range(17):
            temp[:, 0, :, j, :] = sample[:, 0, :,  j, :]*3
        outfile = './data/train_enhance/00' + str(i) + '/'+str(ind)+'_fat.npy'
        np.save(outfile, temp)
for i in range(5):
    path='./data/train/00'+str(i)+'/*.npy'
    for ind, sample_path in enumerate(glob.glob(path)):
        sample = np.load(sample_path)
        temp = sample
        for j in range(17):
            temp[:, 0, :, j, :] = sample[:, 0, :,  j, :]*0.5
        outfile = './data/train_enhance/00' + str(i) + '/'+str(ind)+'_thin.npy'
        np.save(outfile, temp)

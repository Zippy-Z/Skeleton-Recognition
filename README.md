# Skeleton-Recognition
## 项目任务：
通过对人体姿态关键节点序列数据特征提取，对人体姿态序列进行识别和分类

## 数据集及数据集增强：
数据集放在data文件夹中  
data/train文件夹中的数据为老师提供的训练集数据  
data/train_enhance文件夹为经过数据增强的数据集（运行数据集增强代码文件data_enhance.py后会自动创建，由原始训练集数据生成）  
data/test文件夹中的数据为老师提供的测试集数据  
data/train，data/test文件夹中为原始数据，训练数据集有5类数据，总共410个人体关键点序列数据，以.npy形式存在；同理，测试数据集中有5类数据，总共127个数据。  
经过数据集增强后，获得的新的训练集数据样本数为原来的四倍（1640个）。  

## 网络模型：  
RNN模型——RNN.py  
CNN模型——CNN.py  
ST_GCN模型——ST_GCN.py  

## 效果评估：
使用data/test里面5类样本的平均识别率作为指标  
网络模型——平均识别率  
RNN——22.222223%  
CNN——55.555556%  
ST_GCN——27.3504274%  

## 代码运行步骤：  
先运行数据集增强模型——data_enhance.py  
再分别独立运行三个网络模型——RNN.py，CNN.py，ST_GCN.py  


## 需要的模板库
 import torch  
 import random  
 import numpy  
 import glob  
 import os  
 import matplotlib  

## 主要项目负责人  
冯江锴，付旭煜，黄沛吉，胡幸源，朱泽宇

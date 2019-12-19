#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/4 17:08
# @Author  : Peter Zheng
# @File    : default.py
# @Software: PyCharm

from yacs.config import CfgNode as CN

_C = CN()
_C.MODEL = CN()
_C.MODEL.SVM = CN()
_C.MODEL.DNN = CN()
_C.MODEL.DTREE = CN()
_C.MODEL.KNN = CN()
_C.MODEL.NB = CN()
_C.DATA_LOADER = CN()

##此默认配置是针对6m3.csv 数据优化过的##

##################模型总配置################
_C.MODEL.ARCHITECTURES = ["SVM_Model", "DNN_Model", "DTree_Model", "KNN_Model","NaiveBayes_Model","LogisticRegression_Model"]      #模型集合
_C.MODEL.ARCHITECTURE = _C.MODEL.ARCHITECTURES[5]       #模型结构指定
_C.MODEL.SAVE_PATH = "../pre_trained_models"    #模型保存路径配置
_C.MODEL.CROSS_VAL_RESULTS_PATH =  "../cross_val_results/"
###############数据加载器配置##############
_C.DATA_LOADER.TYPE = "Csv_Dataloader"  #数据加载器类型
_C.DATA_LOADER.DATA_PATH = "../dataset/6m3.csv"     #数据路径mrs12m807.csv

###############针对不同模型的配置##########
# SVM 模型配置
_C.MODEL.SVM.DECISION_FUNCTION_SHAPE = 'ovo'
_C.MODEL.SVM.KERNEL = 'rbf'
_C.MODEL.SVM.C = 50
_C.MODEL.SVM.GAMMA = 0.03
_C.MODEL.SVM.PROBABILITY = True

# DNN 模型配置
_C.MODEL.DNN.INPUT = 28
_C.MODEL.DNN.HIDDEN_LAYER_1 = 48
_C.MODEL.DNN.HIDDEN_LAYER_2 = 48
_C.MODEL.DNN.HIDDEN_LAYER_3 = 24
_C.MODEL.DNN.OUTPUT = 2

_C.MODEL.DNN.BATCH_SIZE = 10
_C.MODEL.DNN.LR = 0.005
_C.MODEL.DNN.EPOCH = 300

# DTree 模型

_C.MODEL.DTREE.RANDOM_STATE = 30
_C.MODEL.DTREE.SPLITTER = "random"
_C.MODEL.DTREE.MAX_DEPTH = 5
_C.MODEL.DTREE.MIN_SAMPLES_LEAF = 5
_C.MODEL.DTREE.MIN_SAMPLES_SPLIT = 5

# KNN模型
_C.MODEL.KNN.N_NEIGHBORS = 20
_C.MODEL.KNN.WEIGHTS = 'uniform'
_C.MODEL.KNN.ALGORITHM = 'ball_tree'
_C.MODEL.KNN.METRIC = 'minkowski'

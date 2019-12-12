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

##################模型总配置################
_C.MODEL.ARCHITECTURES = ["SVM_Model", "DNN_Model", "DTree_Model", "KNN_Model","NaiveBayes_Model"]
_C.MODEL.ARCHITECTURE = _C.MODEL.ARCHITECTURES[4]
_C.MODEL.SAVE_PATH = "../pre_trained_models"

###############数据加载器配置##############
_C.DATA_LOADER.TPYE = "Csv_Dataloader"
_C.DATA_LOADER.DATA_PATH = "../dataset/6m3.csv"

###############针对不同模型的配置##########
# SVM 模型配置
_C.MODEL.SVM.DECISION_FUNCTION_SHAPE = 'ovo'
_C.MODEL.SVM.KERNEL = 'linear'
_C.MODEL.SVM.C = 4
_C.MODEL.SVM.GAMMA = 1 / 30
_C.MODEL.SVM.PROBABILITY = True

# DNN 模型配置
_C.MODEL.DNN.INPUT = 28
_C.MODEL.DNN.HIDDEN_LAYER_1 = 56
_C.MODEL.DNN.HIDDEN_LAYER_2 = 56
_C.MODEL.DNN.HIDDEN_LAYER_3 = 28
_C.MODEL.DNN.OUTPUT = 2

_C.MODEL.DNN.BATCH_SIZE = 10
_C.MODEL.DNN.LR = 0.001
_C.MODEL.DNN.EPOCH = 20

# DTree 模型

_C.MODEL.DTREE.RANDOM_STATE = 30
_C.MODEL.DTREE.SPLITTER = "random"
_C.MODEL.DTREE.MAX_DEPTH = 5
_C.MODEL.DTREE.MIN_SAMPLES_LEAF = 10
_C.MODEL.DTREE.MIN_SAMPLES_SPLIT = 10

# KNN模型
_C.MODEL.KNN.N_NEIGHBORS = 20
_C.MODEL.KNN.WEIGHTS = 'uniform'
_C.MODEL.KNN.ALGORITHM = 'ball_tree'
_C.MODEL.KNN.METRIC = 'minkowski'

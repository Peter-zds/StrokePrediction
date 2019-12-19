#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/4 17:27
# @Author  : Peter Zheng
# @File    : model_builder.py
# @Software: PyCharm

from .svm_model import SVM_Model
from .dtree_model import DTree_Model
from .knn_model import KNN_Model
from .nbayes_model import NaiveBayes_Model
from .mlp_model import MLP_Model
from .lr_model import LogisticRegression_Model

_CLASSIFICATION_ARCHITECTURES = {"SVM_Model": SVM_Model, "DNN_Model": MLP_Model, "DTree_Model": DTree_Model,
                                 "KNN_Model": KNN_Model,"NaiveBayes_Model":NaiveBayes_Model,"LogisticRegression_Model":LogisticRegression_Model}

def build_classification_model(cfg):
    model = _CLASSIFICATION_ARCHITECTURES[cfg.MODEL.ARCHITECTURE]
    return model(cfg)

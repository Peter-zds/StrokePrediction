#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/17 20:50
# @Author  : Peter Zheng
# @File    : optimizer_builder.py
# @Software: PyCharm
from .svm_param_selector import SVM_ParamSelector

_OPTIMIZERS = {'SVM_Model':SVM_ParamSelector}
def build_optimizer(cfg,model):
    optimizer = _OPTIMIZERS[cfg.MODEL.ARCHITECTURE]
    return optimizer(model)
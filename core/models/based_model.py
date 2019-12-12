#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/5 10:43
# @Author  : Peter Zheng
# @File    : based_model.py
# @Software: PyCharm
class BasedModel:
    def __init__(self,cfg):
        self.model_name = "BasedModel"
        self.cfg = cfg

    def data_preprocess(self,train_data,train_label,val_data,val_label):
        pass

    def train(self,train_data,train_label,val_data=None, val_label=None):
        pass

    def get_accuracy(self, test_data, test_label):
        pass

    def predict_prob(self, test_data):
        pass

    def predict(self,test_data):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/5 10:43
# @Author  : Peter Zheng
# @File    : based_model.py
# @Software: PyCharm
import os
from sklearn.externals import joblib

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
        path = os.path.join(self.cfg.MODEL.SAVE_PATH, self.model_name + self.get_data_name() + ".m")
        joblib.dump(self.model, path)

    def load_model(self):
        path = os.path.join(self.cfg.MODEL.SAVE_PATH, self.model_name + self.get_data_name() + ".m")
        self.model = joblib.load(path)
    def get_data_name(self):
        file_name = '-'+ self.cfg.DATA_LOADER.DATA_PATH.split('/')[-1].split('.')[0]
        return  file_name
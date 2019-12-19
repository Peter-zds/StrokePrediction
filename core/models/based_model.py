#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/5 10:43
# @Author  : Peter Zheng
# @File    : based_model.py
# @Software: PyCharm
import os
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
class BasedModel:
    def __init__(self,cfg):
        self.model_name = "BasedModel"
        self.cfg = cfg

    def data_preprocess(self,train_data,train_label,val_data,val_label):
        model = SelectKBest(k=12)
        train_data = model.fit_transform(train_data,train_label)
        if val_data is not None:
            val_data=   model.transform(val_data)

        scaler = preprocessing.StandardScaler()
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.fit_transform(val_data)

        # train_data = preprocessing.normalize(train_data, norm='l2')
        # val_data = preprocessing.normalize(val_data, norm='l2')

        # min_max_scaler = preprocessing.MinMaxScaler()
        # train_data = min_max_scaler.fit_transform(train_data)
        # if val_data is not None:
        #     val_data = min_max_scaler.fit_transform(val_data)
        return train_data, train_label, val_data, val_label

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
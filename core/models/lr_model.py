#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/3 11:07
# @Author  : Peter Zheng
# @File    : svm_model.py
# @Software: PyCharm

from .based_model import BasedModel
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest

class LogisticRegression_Model(BasedModel):
    def __init__(self, cfg):
        super(LogisticRegression_Model, self).__init__(cfg)
        self.model = LogisticRegression(solver='liblinear')
        self.model_name = cfg.MODEL.ARCHITECTURE

    def data_preprocess(self,train_data,train_label,val_data,val_label):
        model = SelectKBest(k=12)
        train_data = model.fit_transform(train_data,train_label)
        if val_data is not None:
            val_data=   model.transform(val_data)

        scaler = preprocessing.StandardScaler()
        train_data = scaler.fit_transform(train_data)
        if val_data is not None:
             val_data = scaler.fit_transform(val_data)

        # train_data = preprocessing.normalize(train_data, norm='l2')
        # val_data = preprocessing.normalize(val_data, norm='l2')

        # min_max_scaler = preprocessing.MinMaxScaler()
        # train_data = min_max_scaler.fit_transform(train_data)
        # if val_data is not None:
        #     val_data = min_max_scaler.fit_transform(val_data)
        return train_data, train_label, val_data, val_label

    def train(self, train_data, train_label, val_data=None, val_label=None):
        self.model.fit(train_data, train_label)

    def get_accuracy(self, test_data, test_label):
        return self.model.score(test_data, test_label)

    def predict_prob(self, test_data):
        prob = self.model.predict_proba(test_data)
        return prob[:, 1]

    def predict(self, test_data):
        prediction = self.model.predict(test_data)
        return prediction

#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/3 11:07
# @Author  : Peter Zheng
# @File    : svm_model.py
# @Software: PyCharm

import os
from .based_model import BasedModel
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB

class NaiveBayes_Model(BasedModel):
    def __init__(self, cfg):
        super(NaiveBayes_Model, self).__init__(cfg)
        self.model = GaussianNB()
        self.model_name = cfg.MODEL.ARCHITECTURE

    def data_preprocess(self, train_data, train_label, val_data, val_label):
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

    def save_model(self):
        path = os.path.join(self.cfg.MODEL.SAVE_PATH, self.model_name + ".m")
        joblib.dump(self.model, path)

    def load_model(self):
        path = os.path.join(self.cfg.MODEL.SAVE_PATH, self.model_name + ".m")
        self.model = joblib.load(path)

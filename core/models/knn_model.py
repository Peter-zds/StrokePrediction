#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/3 11:07
# @Author  : Peter Zheng
# @File    : svm_model.py
# @Software: PyCharm

from .based_model import BasedModel
from sklearn.neighbors import KNeighborsClassifier

class KNN_Model(BasedModel):
    def __init__(self, cfg):
        super(KNN_Model, self).__init__(cfg)
        self.model = KNeighborsClassifier(n_neighbors=cfg.MODEL.KNN.N_NEIGHBORS,
                                          weights=cfg.MODEL.KNN.WEIGHTS,
                                          algorithm=cfg.MODEL.KNN.ALGORITHM,
                                          metric=cfg.MODEL.KNN.METRIC)
        self.model_name = cfg.MODEL.ARCHITECTURE

    def data_preprocess(self, train_data, train_label, val_data, val_label):
        return train_data, train_label, val_data, val_label

    def train(self, train_data, train_label, val_data=None, val_label=None):
        self.model.fit(train_data,train_label)

    def get_accuracy(self, test_data, test_label):
        return self.model.score(test_data, test_label)

    def predict_prob(self, test_data):
        prob = self.model.predict_proba(test_data)
        return prob[:, 1]

    def predict(self, test_data):
        prediction = self.model.predict(test_data)
        return prediction


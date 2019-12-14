#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/3 11:07
# @Author  : Peter Zheng
# @File    : svm_model.py
# @Software: PyCharm

from .based_model import BasedModel
from sklearn import tree

class DTree_Model(BasedModel):
    def __init__(self, cfg):
        super(DTree_Model, self).__init__(cfg)
        self.model = tree.DecisionTreeClassifier(random_state=cfg.MODEL.DTREE.RANDOM_STATE
                                  ,splitter=cfg.MODEL.DTREE.SPLITTER
                                  ,max_depth=cfg.MODEL.DTREE.MAX_DEPTH
                                  ,min_samples_leaf=cfg.MODEL.DTREE.MIN_SAMPLES_LEAF
                                  ,min_samples_split=cfg.MODEL.DTREE.MIN_SAMPLES_SPLIT)
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


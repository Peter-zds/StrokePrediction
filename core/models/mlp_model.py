#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/18 9:16
# @Author  : Peter Zheng
# @File    : mlp_model.py
# @Software: PyCharm

from .based_model import BasedModel
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest

class MLP_Model(BasedModel):
    def __init__(self, cfg):
        super(MLP_Model, self).__init__(cfg)
        self.model = MLPClassifier(hidden_layer_sizes=(cfg.MODEL.DNN.HIDDEN_LAYER_1,cfg.MODEL.DNN.HIDDEN_LAYER_2,cfg.MODEL.DNN.HIDDEN_LAYER_3)
                                   ,activation='relu',batch_size=cfg.MODEL.DNN.BATCH_SIZE,learning_rate='constant',
                                   learning_rate_init = cfg.MODEL.DNN.LR,max_iter=cfg.MODEL.DNN.EPOCH
        )
        self.model_name = cfg.MODEL.ARCHITECTURE

    def data_preprocess(self,train_data,train_label,val_data,val_label):

        train_data = preprocessing.normalize(train_data, norm='l2')
        if val_data is not None:
            val_data = preprocessing.normalize(val_data, norm='l2')

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
#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/3 11:08
# @Author  : Peter Zheng
# @File    : csv_dataloader.py
# @Software: PyCharm

import pandas
import numpy as np

from sklearn.model_selection import train_test_split


class Csv_Dataloader:
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train

    def read_csv_as_numpy(self, path: str):
        f = open(path)
        data_frame = pandas.read_csv(f, dtype=np.float32, header=0)
        data = data_frame.values
        return data

    def split_data_lable(self, dataset):
        data = dataset[:, :-2]
        label_value = dataset[:, -2]
        label_classify = dataset[:, -1]
        return data, label_value, label_classify

    def load_data(self):
        dataset = self.read_csv_as_numpy(self.cfg.DATA_LOADER.DATA_PATH)
        data, label_value, label_classify = self.split_data_lable(dataset)
        return train_test_split(data, label_classify, test_size=0.2, random_state=42)

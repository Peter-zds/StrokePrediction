#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/4 18:10
# @Author  : Peter Zheng
# @File    : dataloader_builder.py
# @Software: PyCharm
from .csv_dataloader import Csv_Dataloader

_DATA_LOADERS = {"Csv_Dataloader":Csv_Dataloader}
def make_dataloader(cfg,is_trian = True):
    data_loader =  _DATA_LOADERS[cfg.DATA_LOADER.TYPE]
    return data_loader(cfg,is_trian)
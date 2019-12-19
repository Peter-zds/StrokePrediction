#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/17 21:03
# @Author  : Peter Zheng
# @File    : param_optimize.py
# @Software: PyCharm
import sys
sys.path.append('../')
import argparse
from core.config import cfg
from core.dataloader import make_dataloader
from core.model_optimize.optimizer_builder import build_optimizer
from core.models.model_builder import build_classification_model
def optimize(cfg):
    model =build_classification_model(cfg)
    dataloader = make_dataloader(cfg)
    optimizer = build_optimizer(cfg,model)
    data, label_value, label_classify = dataloader.split_data_lable()
    optimizer.select_param(data=data, label=label_classify)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    if args.config_file is not '':
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    optimize(cfg)
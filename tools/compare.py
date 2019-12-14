#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/3 11:06
# @Author  : Peter Zheng
# @File    : train.py
# @Software: PyCharm
import argparse
import sys
sys.path.append('../')
from core.models import build_classification_model
from core.config import cfg
from core.dataloader import make_dataloader
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

def compare(cfg):
    data_loader = make_dataloader(cfg,True)
    train_data,test_data,train_label,test_label = data_loader.load_data()
    cfg_cp = cfg.clone()
    for m in cfg.MODEL.ARCHITECTURES:
        cfg_cp.MODEL.ARCHITECTURE = m
        model = build_classification_model(cfg_cp)
        model.load_model()
        pre_train_data, pre_train_label, pre_test_data, pre_test_label = model.data_preprocess(train_data, train_label, test_data, test_label)
        acc = model.get_accuracy(pre_test_data,pre_test_label)
        prediction = model.predict_prob(pre_test_data)
        fpr, tpr, thresholds = roc_curve(pre_test_label,prediction)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label= model.model_name + '(area = {0:.2f})'.format(roc_auc))
        print(model.model_name+"测试集正确率为：" + "{:.2f}".format(acc * 100) + "%")
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

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
    compare(cfg)
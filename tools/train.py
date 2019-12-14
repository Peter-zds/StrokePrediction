#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/3 11:06
# @Author  : Peter Zheng
# @File    : train.py
# @Software: PyCharm
import sys
sys.path.append('../')
import argparse
from core.models import build_classification_model
from core.config import cfg
from core.dataloader import make_dataloader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def train(cfg):
    model = build_classification_model(cfg)
    data_loader = make_dataloader(cfg, True)
    train_data, val_data, train_label, val_label = data_loader.load_data()
    train_data, train_label, val_data, val_label = model.data_preprocess(train_data, train_label, val_data, val_label)
    model.train(train_data, train_label, val_data, val_label)
    acc = model.get_accuracy(val_data,val_label)
    prediction = model.predict_prob(val_data)
    fpr, tpr, thresholds = roc_curve(val_label,prediction)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(1, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    print("测试集正确率为：" + "{:.2f}".format(acc * 100) + "%")
    plt.show()
    model.save_model()


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
    train(cfg)

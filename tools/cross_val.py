#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/16 21:19
# @Author  : Peter Zheng
# @File    : cross_val.py
# @Software: PyCharm

import sys
sys.path.append('../')
from sklearn.model_selection import KFold
import numpy as np
from scipy import interp
import argparse
from core.models import build_classification_model
from core.config import cfg
from core.dataloader import make_dataloader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def cross_val(cfg):
    KF = KFold(n_splits = 5,shuffle=True)
    tprs = []
    aucs = []
    accs = []
    mean_fpr = np.linspace(0, 1, 100)

    model = build_classification_model(cfg)
    data_loader = make_dataloader(cfg, True)
    data, label_value, label_classify = data_loader.split_data_lable()
    for i,(train_idx, val_idx) in enumerate(KF.split(data)):
        train_data,val_data = data[train_idx],data[val_idx]
        train_label,val_label = label_classify[train_idx],label_classify[val_idx]
        train_data, train_label, val_data, val_label = model.data_preprocess(train_data, train_label, val_data, val_label)
        model.train(train_data, train_label, val_data, val_label)
        acc = model.get_accuracy(val_data, val_label)
        accs.append(acc)
        prediction = model.predict_prob(val_data)
        fpr, tpr, thresholds = roc_curve(val_label, prediction)
        roc_auc = auc(fpr, tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
        print(model.model_name + "测试集正确率为：" + "{0:.3f}".format(acc * 100) + "%")
        plt.plot(mean_fpr, tprs[-1], lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))

    mean_acc = np.mean(accs)
    print(model.model_name + "测试集平均正确率为：" + "{0:.3f}".format(mean_acc * 100) + "%")
    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.3f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model.model_name+':ROC')
    plt.legend(loc='lower right')
    plt.show()
    np.savez(cfg.MODEL.CROSS_VAL_RESULTS_PATH + model.model_name +
             '-'+ cfg.DATA_LOADER.DATA_PATH.split('/')[-1].split('.')[0] + '.npz',
             mean_fpr=mean_fpr,tprs=tprs,aucs=aucs,accs=accs)

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
    cross_val(cfg)
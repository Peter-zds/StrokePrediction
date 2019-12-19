#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/19 14:36
# @Author  : Peter Zheng
# @File    : statistic.py
# @Software: PyCharm

import sys
sys.path.append('../')
import os
import numpy as np
import argparse
from core.config import cfg
import matplotlib.pyplot as plt

def statistic(cfg):
    model_names = cfg.MODEL.ARCHITECTURES
    mean_tprs = []
    mean_aucs = []
    mean_accs = []
    mean_fpr = []
    temp_names = []
    for name in model_names:
        file_name = cfg.MODEL.CROSS_VAL_RESULTS_PATH + name + '-'+ cfg.DATA_LOADER.DATA_PATH.split('/')[-1].split('.')[0] + '.npz'
        if os.path.exists(file_name):
            arr = np.load(file_name)
            mean_fpr = arr['mean_fpr']
            tprs = arr['tprs']
            aucs = arr['aucs']
            accs = arr['accs']
            temp_names.append(name)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            mean_acc = np.mean(accs)

            mean_tprs.append(mean_tpr)
            mean_aucs.append(mean_auc)
            mean_accs.append(mean_acc)
            for i,(tpr, roc_auc) in enumerate(zip(tprs,aucs)):
                plt.plot(mean_fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.3f)' % (i, roc_auc))
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
            plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.3f)' % mean_auc, lw=2, alpha=.8)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(name + ':ROC')
            plt.legend(loc='lower right')
            plt.show()
    for (mean_tpr,mean_auc,temp_name) in  zip(mean_tprs,mean_aucs,temp_names):
        plt.plot(mean_fpr, mean_tpr, lw=1, alpha=1, label= temp_name+ '(area=%0.3f)' % mean_auc)
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Statistic ROC Curve')
    plt.legend(loc='lower right')
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
    cfg.freeze()
    statistic(cfg)
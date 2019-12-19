#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/17 20:20
# @Author  : Peter Zheng
# @File    : svm_param_selector.py
# @Software: PyCharm

from sklearn.model_selection import GridSearchCV

class SVM_ParamSelector:
    def __init__(self,model):
        self.model = model
        self.param_grid = {'C':range(1,30),'gamma':[0.5,0.3,0.1,0.03,0.01,0.001],'kernel':['linear','rbf']}
        self.param_selector = GridSearchCV(model.model,self.param_grid,scoring='roc_auc',cv=5)

    def select_param(self,data,label):
        train_data, train_label, val_data, val_label = self.model.data_preprocess(data, label, None,None)
        self.param_selector.fit(train_data, train_label)
        print('网格搜索-度量记录：', self.param_selector.cv_results_)  # 包含每次训练的相关信息
        print('网格搜索-最佳度量值:', self.param_selector.best_score_)  # 获取最佳度量值
        print('网格搜索-最佳参数：', self.param_selector.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
        print('网格搜索-最佳模型：', self.param_selector.best_estimator_)  # 获取最佳度量时的分类器模型
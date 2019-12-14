#!/usr/bin/python3
# -*-coding:utf-8 -*-
# @Time    : 2019/12/5 16:28
# @Author  : Peter Zheng
# @File    : dnn_model.py
# @Software: PyCharm

from .based_model import BasedModel
import torch
from torch import nn
import torch.utils.data as Data
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt


class DNN_Model(BasedModel):
    def __init__(self, cfg):
        super(DNN_Model, self).__init__(cfg)
        self.model_name = cfg.MODEL.ARCHITECTURE
        self.model = nn.Sequential(
            nn.Linear(cfg.MODEL.DNN.INPUT, cfg.MODEL.DNN.HIDDEN_LAYER_1),
            nn.Linear(cfg.MODEL.DNN.HIDDEN_LAYER_1, cfg.MODEL.DNN.HIDDEN_LAYER_2),
            nn.Linear(cfg.MODEL.DNN.HIDDEN_LAYER_2, cfg.MODEL.DNN.HIDDEN_LAYER_3),
            nn.ReLU(),
            nn.Linear(cfg.MODEL.DNN.HIDDEN_LAYER_3, cfg.MODEL.DNN.OUTPUT),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.MODEL.DNN.LR)  # optimize all cnn parameters
        self.loss_func = nn.CrossEntropyLoss()

    def data_preprocess(self, train_data, train_label, val_data, val_label):
        if train_data is not None and not torch.is_tensor(train_data):
            train_data = torch.from_numpy(train_data)
            train_label = torch.from_numpy(train_label).long()
        if val_data is not None and not torch.is_tensor(val_data):
            val_data = torch.from_numpy(val_data)
            val_label = torch.from_numpy(val_label).long()
        return train_data, train_label, val_data, val_label

    def train(self, train_data, train_label, val_data=None, val_label=None):
        dataset = Data.TensorDataset(train_data, train_label)
        data_loader = Data.DataLoader(dataset=dataset, batch_size=self.cfg.MODEL.DNN.BATCH_SIZE, shuffle=True,
                                      num_workers=2)
        losses = []
        for epoch in range(self.cfg.MODEL.DNN.EPOCH):
            for step, (batch_x, batch_y) in enumerate(data_loader):
                output = self.model(batch_x)
                loss = self.loss_func(output, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss)
            if val_data is not None and epoch % 2 == 0:
                with  torch.no_grad():
                    plt.plot(losses)
                    plt.ylim([-0.05, 10])
                    plt.show()
                    accuracy = self.get_accuracy(val_data, val_label)
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(),
                          '| test accuracy: %.2f' % accuracy)
                    if accuracy >= 0.78:
                        break

    def get_accuracy(self, test_data, test_label):
        test_output = self.model(test_data)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        test_label = test_label.data.numpy()
        correct_num = (pred_y == test_label).astype(int).sum()
        accuracy = float(correct_num) / float(test_label.size)
        return accuracy

    def predict_prob(self, test_data):
        test_output = self.model(test_data)
        prob = F.softmax(test_output,1)
        return prob.data.numpy()[:,1]
    def predict(self, test_data):
        test_output = self.model(test_data)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        return pred_y

    def save_model(self):
        path = os.path.join(self.cfg.MODEL.SAVE_PATH, self.model_name + self.get_data_name()+".pkl")
        torch.save(self.model,path)
    def load_model(self):
        path = os.path.join(self.cfg.MODEL.SAVE_PATH, self.model_name + self.get_data_name()+".pkl")
        self.model = torch.load(path)
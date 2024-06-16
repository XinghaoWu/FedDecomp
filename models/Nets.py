#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn
import torch.nn.functional as F

class CNNMnist_NoBN_easy(nn.Module):
    def __init__(self, args):
        super(CNNMnist_NoBN_easy, self).__init__()
        self.feature1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 16, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.feature2 = nn.Sequential(
            nn.Linear(4096, 2048),
            # nn.BatchNorm1d(3072),
            nn.ReLU(),

            nn.Dropout(),

            nn.Linear(2048, 1024),
            # nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.C = nn.Linear(1024, args.num_classes)

        # self.C = nn.Linear(2048*2, args.num_classes)

    def forward(self, x):
        f1 = self.feature1(x)
        f1 = f1.view(-1, f1.shape[1] * f1.shape[2] * f1.shape[3])
        f1 = self.feature2(f1)
        feature = copy.deepcopy(f1)

        c2 = self.C(f1)
        c3 = F.log_softmax(c2, dim=1)
        return None, None, feature, c3




class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return F.log_softmax(x, dim=1)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNMnist_Transfer(nn.Module):
    def __init__(self, args):
        super(CNNMnist_Transfer, self).__init__()
        self.feature1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.feature2 = nn.Sequential(
            nn.Linear(8192, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(),

            nn.Dropout(),

            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.C = nn.Linear(2048, args.num_classes)

    def forward(self, x):
        f1 = self.feature1(x)
        f1 = f1.view(-1, f1.shape[1] * f1.shape[2] * f1.shape[3])
        f2 = self.feature2(f1)
        c1 = self.C(f2)
        c1 = F.log_softmax(c1, dim=1)
        return c1
        # return f1, c1




class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    test_model = CNNMnist_Transfer(3, 10)
    test_img = torch.ones([1, 3, 32, 32])
    f2, c1 = test_model(test_img)
    # test_model.train()
    # para_dict = test_model.named_parameters()
    # para_dict["conv1.weight"].requires_grad = True
    # print(type(para_dict))
    # for name, para in para_dict:
    #     print(name, para.requires_grad)
    # test_model.conv1.requires_grad_()
    # print(para_dict)

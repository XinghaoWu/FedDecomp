#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg_list = []
    for domain_idx in w:
        w_avg = copy.deepcopy(w[domain_idx][0])
        for k in w_avg.keys():
            for i in range(1, len(w[domain_idx])):
                w_avg[k] += w[domain_idx][i][k]
            w_avg[k] = torch.div(w_avg[k], len(w[domain_idx]))
        w_avg_list.append(w_avg)

    w_avg = copy.deepcopy(w_avg_list[0])
    for k in w_avg.keys():
        for i in range(1, len(w_avg_list)):
            w_avg[k] += w_avg_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_avg_list))
    # distance(w_avg_list[0], w_avg_list[1])
    # distance(w_avg, w_avg_list[0])
    return w_avg


def distance(w1, w2):
    for k in w1.keys():
        print((w1[k] - w2[k])**2)
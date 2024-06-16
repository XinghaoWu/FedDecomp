#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=40, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5
                        , help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=100, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--optimizer', type=str, default='sgd', help='type of optimizer')
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # dataset argument
    parser.add_argument('--train_num', type=int, default=500, help='number of training samples for training')
    parser.add_argument('--test_num', type=int, default=100, help='number of testing samples for training')
    parser.add_argument('--scale', type=int, default=32, help='image size after loading')
    parser.add_argument('--dataset', type=str, default='cifar-100', choices=['cifar', 'cifar-100', 'tinyimagenet'] ,help="name of dataset")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--noniid', type=str, default='dirichlet',
                        help='Default set to pathological Non-IID.')
    parser.add_argument('--alpha', type=float, default=1.0, help='the degree of imbalance')
    parser.add_argument('--random', type=bool, default=True, help='whether random choose class for each client')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                                non-i.i.d setting (use 0 for equal splits)')

    # model arguments
    parser.add_argument('--model', type=str, default='resnet10', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")


    # other arguments
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--policy', type=int, default=3, choices=[1, 3], help='global training policy')
    parser.add_argument('--save_model', type=int, default=0, help='Whether to save models. Set 0 to not save')
    parser.add_argument('--compare', type=str, default='test',
                        help='repeat tag')
    parser.add_argument('--repeat', default=1, type=int, help='repeat times')
    parser.add_argument('--method', type=int, default=0, choices=[0, 1], help='0: no finetune; 1: with finetune')
    parser.add_argument('--Conv_r', type=float, default=0.8, help='rank of Conv layer')
    parser.add_argument('--Linear_r', type=int, default=40, help='rank of FC layer')
    parser.add_argument('--local_p_ep', type=int, default=2, help='epoch of personalized part')
    args = parser.parse_args()
    args.alpha = args.alpha if args.noniid == 'dirichlet' else int(args.alpha)
    return args

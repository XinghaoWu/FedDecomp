import numpy as np
from torchvision import datasets, transforms
import csv
import torch
from utils.options import args_parser
from src.client import Client
from src.server import Server
from models.Nets import CNNCifar, MLP
from models.resnet import resnet18, resnet8, resnet10
from utils.util import set_logger
import random
import torchvision.models as torch_model
import models.LoraResnet1 as lora_model
# np.set_printoptions(threshold=1000000)


def exp_parameter(args):
    print(f'Communication Rounds: {args.epochs}')
    print(f'Client Number: {args.num_users}')
    print(f'Local Epochs: {args.local_ep}')
    print(f'Local Batch Size: {args.local_bs}')
    print(f'Learning Rate: {args.lr}')
    print(f'Policy: {args.policy}')
    print(f'Communication Rounds: {args.epochs}')
    print(f'noniid: {args.noniid}')
    print(f'alpha: {args.alpha}')
    print(f'seed: {args.seed}')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, logger=None):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    accList = []
    for t in range(args.repeat):    # repeat args.repeat times
        if args.seed == -1: # if seed is not specified
            args.seed = np.random.randint(0,10000)  # random select a seed
        setup_seed(args.seed)
        logFileName = f'./log/FedDecomp_dataset{args.dataset}_model{args.model}_frac{args.frac}_policy{args.policy}_method{args.method}_round{args.epochs}_epoch{args.local_ep}_pep{args.local_p_ep}_ConvR{args.Conv_r}_LinearR{args.Linear_r}_optimizer{args.optimizer}_lr{args.lr}_{args.noniid}_{args.alpha}_trainnum{args.train_num}_seed{args.seed}_compare({args.compare}).log'
        logger = set_logger(logFileName)

        if args.dataset == 'cifar':
            args.num_classes = 10
            if args.model == 'resnet8':
                local_model = lora_model.resnet8(num_labels=args.num_classes, Conv_r=args.Conv_r,
                                                 Linear_r=args.Linear_r).to(device)
            else:
                raise NotImplementedError
        elif args.dataset == 'cifar-100':
            args.num_classes = 100
            if args.model == 'resnet8':
               local_model = lora_model.resnet8(num_labels=args.num_classes, Conv_r=args.Conv_r, Linear_r=args.Linear_r).to(device)
            elif args.model == 'resnet10':
                local_model = lora_model.resnet10(num_labels=args.num_classes, Conv_r=args.Conv_r, Linear_r=args.Linear_r).to(device)
            else:
                raise NotImplementedError
        elif args.dataset == 'tinyimagenet':
            args.num_classes = 200
            if args.model == 'resnet8':
                local_model = lora_model.resnet8(num_labels=args.num_classes, Conv_r=args.Conv_r,
                                                 Linear_r=args.Linear_r).to(device)
            elif args.model == 'resnet10':
                local_model = lora_model.resnet10(num_labels=args.num_classes, Conv_r=args.Conv_r,
                                                  Linear_r=args.Linear_r).to(device)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        print(f'Model Structure: {local_model}')
        logger.info(f'Model Structure: {local_model}')
        server = Server(device, local_model, args, logger=logger)
        server.train()
        print('Best:', server.best_accuracy_global_after)
        logger.info(f'Best: {server.best_accuracy_global_after}')
        accList.append(max(server.test_acc))
        args.seed = -1
    print(f'Repeat {args.repeat} times, mean:{np.mean(accList)}, std:{np.std(accList)}')




if __name__ == '__main__':
    args = args_parser()
    args.verbose = 0
    exp_parameter(args)
    train(args)



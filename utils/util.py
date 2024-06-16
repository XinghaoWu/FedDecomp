import torch
import numpy as np
import logging

def model_distance(model1, model2):
    res = 0
    # weight_num = sum(w.numel() for w in model1.parameters())
    for k in model1.keys():
        res += torch.sum((model1[k] - model2[k]) ** 2)
    return torch.sqrt(res)

def model_distance_manhattan(model1, model2):
    res = 0
    for k in model1.keys():
        res += torch.sum(torch.abs(model1[k] - model2[k]))
    return res

def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)

    return params

def weight_flatten_by_name(model, name):
    params = []
    for u in model.named_parameters():
        if name not in u[0]:
            params.append(u[1].view(-1))
    params = torch.cat(params)
    return params


def set_logger(file_path = 'log.txt', handle = 1):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create Handler
    # type 1: file handler
    # type 2: stream handler
    if handle == 1:
        log_handler = logging.FileHandler(file_path, mode='w', encoding='UTF-8')
    elif handle == 2:
        log_handler = logging.StreamHandler()
    else:
        log_handler = logging.FileHandler(file_path, mode='w', encoding='UTF-8')

    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)

    # Add to logger
    logger.addHandler(log_handler)

    return logger

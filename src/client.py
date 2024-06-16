import copy
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
np.set_printoptions(edgeitems=30)
torch.set_printoptions(edgeitems=30)
import loralib
import time

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class Client:
    def __init__(self, device, local_model, train_dataset, test_dataset, train_idxs, test_idxs, args, index, logger=None):
        self.device = device
        self.args = args
        self.index = index
        self.local_model = local_model
        self.logger = logger

        self.trainingLoss = None
        self.testingLoss = None
        self.testingAcc = None


        self.trainloader, self.validloader, self.testloader, self.trainloader_full = self.train_val_test(
            train_dataset, list(train_idxs), test_dataset, list(test_idxs))

        # define Loss function
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, train_dataset, train_idxs, test_dataset, test_idxs):
        """
        Returcns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        trainloader = DataLoader(DatasetSplit(train_dataset, train_idxs),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = None
        testloader = DataLoader(DatasetSplit(test_dataset, test_idxs),
                                batch_size=int(len(test_idxs) / 10), shuffle=False)
        trainloader_full = DataLoader(DatasetSplit(train_dataset, train_idxs), batch_size=len(train_idxs), shuffle=False)

        return trainloader, validloader, testloader, trainloader_full

    def train(self):
        self.local_model.to(self.device)
        self.local_model.train()
        epoch_loss = []

        # define optimizer
        if self.args.optimizer == 'sgd':
            weights = dict(self.local_model.named_parameters())
            lora_weights, non_lora_weights = [], []
            for k in weights.keys():
                if 'lora_A' in k or 'lora_B' in k:
                    lora_weights.append(weights[k])
                else:
                    non_lora_weights.append(weights[k])
            self.optimizer_lora = torch.optim.SGD(lora_weights, lr=self.args.lr, momentum=self.args.momentum)
            self.optimizer_nonlora = torch.optim.SGD(non_lora_weights, lr=self.args.lr, momentum=self.args.momentum)
        else:
            raise NotImplementedError

        start_time = time.time()
        # train personalized part
        loralib.mark_only_lora_as_trainable(self.local_model)
        for iter in range(self.args.local_p_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                log_probs = self.local_model(images)
                loss = self.criterion(log_probs, labels.long())
                self.optimizer_lora.zero_grad()
                loss.backward()

                self.optimizer_lora.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # train shared part
        loralib.mark_only_weight_as_trainable(self.local_model)
        for iter in range(self.args.local_ep - self.args.local_p_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                log_probs = self.local_model(images)
                loss = self.criterion(log_probs, labels.long())
                self.optimizer_nonlora.zero_grad()
                loss.backward()

                self.optimizer_nonlora.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.trainingLoss = sum(epoch_loss) / len(epoch_loss)
        end_time = time.time()
        self.local_model.to('cpu')
        return sum(epoch_loss) / len(epoch_loss), end_time - start_time



    def inference(self, mode='all'):
        self.local_model.to(self.device)
        self.local_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        count = 1
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # inference
                outputs = self.local_model(images, mode)
                batch_loss = self.criterion(outputs, labels.long())
                loss += batch_loss.item()

                # prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels.long())).item()
                total += len(labels)
                count += 1

        accuracy = correct / total
        loss = loss / count
        self.testingAcc, self.testingLoss = accuracy, loss
        self.local_model.to('cpu')
        return accuracy, loss

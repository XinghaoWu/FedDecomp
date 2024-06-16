import copy
import torch
from torchvision import datasets, transforms
from utils.sampling import mnist_iid
from utils.sampling import cifar_iid
from utils.sampling import dirichlet_noniid, pathological_noniid

import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class TinyImageNet(Dataset):
    def __init__(self, transform=None, is_train=True):
        self.data_dir = r"./data/tiny-imagenet-200/"
        with open(self.data_dir + 'wnids.txt', 'r') as f:
            wnids = [x.strip() for x in f]

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}
        label_to_wnid = {v: k for k, v in wnid_to_label.items()}

        self.img_files = []
        self.targets = []
        if is_train:
            for k, v in wnid_to_label.items():
                images_path = self.data_dir + 'train' + '/' + str(k) + '/images/'
                images_name = os.listdir(images_path)
                for name in images_name:
                    self.img_files.append(images_path + name)
                    self.targets.append(v)
        else:
            with open(os.path.join(self.data_dir, 'val', 'val_annotations.txt'), 'r') as f:
                img_files = []
                val_wnids = []
                for line in f:
                    img_file, wnid = line.split('\t')[:2]
                    img_files.append(img_file)
                    val_wnids.append(wnid)
                self.img_files = [self.data_dir + 'val/images/' + item for item in img_files]
                self.targets = [wnid_to_label[wnid] for wnid in val_wnids]

        self.transform = transform

    def __getitem__(self, index):

        with open(self.img_files[index], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            # img = np.array(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index]

    def __len__(self):
        return len(self.img_files)

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            train_user_groups, test_user_groups = cifar_iid(train_dataset, test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                if args.noniid == 'pathological':
                    train_user_groups, test_user_groups = pathological_noniid(train_dataset, test_dataset, args.num_users,
                                                                         args.alpha, args.seed, args, random=args.random)
                else:
                    train_user_groups, test_user_groups = dirichlet_noniid(train_dataset, test_dataset, args.num_users,
                                                                         args.alpha, args.seed, args)
                    for i in range(2):
                        print("dirichlet noniid")


    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        else:
            data_dir = './data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=False, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            train_user_groups, test_user_groups = mnist_iid(train_dataset, test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                pass
                # user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                if args.noniid == 'pathological':
                    train_user_groups, test_user_groups = pathological_noniid(train_dataset, test_dataset, args.num_users,
                                                                     args.alpha, args.seed, args, random=args.random)

                else:
                    train_user_groups, test_user_groups = dirichlet_noniid(train_dataset, test_dataset, args.num_users,
                                                                           args.alpha, args.seed, args)
                    for i in range(2):
                        print("dirichlet noniid")
    elif args.dataset == 'cifar-100':
        data_dir = './data/cifar-100/'
        apply_transform1 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        apply_transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=apply_transform1)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=apply_transform2)
        if args.iid:
            raise NotImplementedError
        else:
            if args.noniid == 'pathological':
                train_user_groups, test_user_groups = pathological_noniid(train_dataset, test_dataset, args.num_users,
                                                                         args.alpha, args.seed, args, random=args.random)
            else:
                train_user_groups, test_user_groups = dirichlet_noniid(train_dataset, test_dataset, args.num_users,
                                                                       args.alpha, args.seed, args)
                for i in range(2):
                    print("dirichlet noniid")
    elif args.dataset == 'tinyimagenet':
        apply_transform1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        apply_transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225], inplace=True)
             ])

        train_dataset = TinyImageNet(transform=apply_transform1, is_train=True)
        test_dataset = TinyImageNet(transform=apply_transform2, is_train=False)
        if args.iid:
            raise NotImplementedError
        else:
            if args.noniid == 'pathological':
                train_user_groups, test_user_groups = pathological_noniid(train_dataset, test_dataset, args.num_users,
                                                                         args.alpha, args.seed, args, random=args.random)
            else:
                train_user_groups, test_user_groups = dirichlet_noniid(train_dataset, test_dataset, args.num_users,
                                                                       args.alpha, args.seed, args)
                for i in range(2):
                    print("dirichlet noniid")
    else:
        raise  NotImplementedError
    return train_dataset, test_dataset, train_user_groups, test_user_groups


import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import torch
import torchvision.datasets as dset
import torch.utils.data as data

def cifar10_dataloader(transform_train, transform_test, train_batch=128, test_batch=100, shuffle_train=True, shuffle_test=False):
    '''
    Args:
        transform_train - transforms.Compose([        ])
        transform_test - transforms.Compose([        ])
        train_batch - int defaults to be 128
        test_batch - int defaults to be 100.
        shuffle_train - bool defaults to be True
        shuffle_test - bool defaults to be Test
    Return : 
        trainloader, testloader
    '''
    trainset = dset.CIFAR10(root='/data_large/readonly/', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=shuffle_train, num_workers=4)

    testset = dset.CIFAR10(root='/data_large/readonly/', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=shuffle_test, num_workers=4)

    return trainloader, testloader


# Use the ResNet18 on Cifar-10
import os
import sys

from src import device

import torch
import torchvision.transforms as transforms
from src.deeplearning.htru1 import HTRU1

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)
# set hyperparameter
BATCH_SIZE = 128

trainloader=None
testloader=None


def load_data():
    global trainloader
    global testloader
    # convert data to a normalized torch.FloatTensor
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # randomly flip and rotate
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # choose the training and test datasets

    trainset = HTRU1(root=ROOT_PATH+'/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = HTRU1(root=ROOT_PATH+'/data', train=False, download=True, transform=transform_test)
    size = len(testset.targets)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=size, shuffle=False, num_workers=2)


def load_model(path):
    model = torch.load(path, map_location=device)
    print("model loaded")
    return model

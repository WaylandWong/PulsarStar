# Use the ResNet18 on plusarstar
import os
import sys

from torch.utils.data import Dataset, DataLoader

from src import device

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from src.utils.dataset import data_preprocess


# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainloader=None
testloader=None

BATCH_SIZE = 128

class PulsarStarDataset(Dataset):

    def __init__(self, train=True):
        x_train, y_train, x_test, y_test = data_preprocess()

        if train:
            self.x_train = torch.tensor(x_train.values, dtype=torch.float32, device=device)
            self.y_train = torch.tensor(y_train.values, dtype=torch.long, device=device)
        else:
            self.x_train = torch.tensor(x_test.values, dtype=torch.float32, device=device)
            self.y_train = torch.tensor(y_test.values, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

def load_data():
    global trainloader
    global testloader
    train_dataset = PulsarStarDataset(train=True)
    test_dataset = PulsarStarDataset(train=False)

    size = test_dataset.y_train.size()[0]

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=size, shuffle=False, num_workers=2, pin_memory=True)


def load_model(path):
    model = torch.load(path, map_location=device)
    print("model loaded")
    return model

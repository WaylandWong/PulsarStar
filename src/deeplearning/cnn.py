import torch
from torch import nn
from torchinfo import summary

from src.deeplearning import ann
from src import device


# Improving the ANN using Grid Search and Dropout regularization
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        outputs = self.linear_relu_stack(x)
        return outputs


def info(num_classes=10):
    net = NeuralNetwork(num_classes=num_classes).to(device)
    summary(net, (1, 8))


def train(trainloader, testloader, outpath, num_classes=10):
    net = NeuralNetwork(num_classes=num_classes).to(device)
    ann.train(net, trainloader, testloader, outpath, weights=[1, 2])


def pred(model, testloader):
    # net = NeuralNetwork(num_classes=num_classes).to(device)
    # net.load_state_dict(model)
    ann.pred(model, device, testloader, show_confusion=True)

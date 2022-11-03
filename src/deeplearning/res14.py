import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from src import deeplearning
from src.deeplearning import ann


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 8
        self.layer1 = self.make_layer(ResidualBlock, 8, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 16, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 16, 1, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 16, 1, stride=2)
        self.fc = nn.Linear(16, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x[:, None, :]
        out = out.permute(0, 2, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet14(num_classes):
    return ResNet(ResidualBlock, num_classes=num_classes)


def info(num_classes=10):
    net = ResNet14(num_classes=num_classes).to(device)
    summary(net, (1, 8))


def train(trainloader, testloader, outpath, num_classes=2):
    # define ResNet14
    net = ResNet14(num_classes=num_classes).to(device)
    ann.train(net, trainloader, testloader, outpath)


def pred(model, testloader):
    # net = ResNet14().to(device)
    # net.load_state_dict(model)
    ann.pred(model, device, testloader, show_confusion=True)


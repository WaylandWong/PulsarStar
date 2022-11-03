from torch import nn, optim

from src import deeplearning, device
import torch

from src.utils.performance import performance
from src.utils.roc import roc, history

# set hyperparameter
pre_epoch = 0
EPOCH = 10
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9

loss_history = []
accuracy_history = []
precision_history = []
recall_history = []

test_loss_history = []
test_accuracy_history = []
test_precision_history = []
test_recall_history = []


def train(net, trainloader, testloader, outpath, weights=None):
    # define loss function & optimizer
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=5e-4)

    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        # sum_loss = 0.0
        # correct = 0.0
        # total = 0.0
        for i, data in enumerate(trainloader, 0):
            # prepare dataset
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward & backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            loss_num = loss.item()
            loss_history.append(loss_num)
            # sum_loss += loss_num
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += predicted.eq(labels.data).sum()
            # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
            #       % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

            # get the ac with testdataset in each epoch
            # if i % 1 == 0:
        print('Waiting Test...')
        pred(net, trainloader, train=True, show_confusion=False)
        pred(net, testloader, train=False, show_confusion=False)

    print('Train has finished, total epoch is %d' % EPOCH)
    pred(net, trainloader, train=True, show_confusion=False)
    pred(net, testloader, train=False, show_confusion=True)
    torch.save(net, outpath)
    data = {
        'loss': loss_history,
        'fn': accuracy_history,
        'val_fn': test_accuracy_history,
        'precision': precision_history,
        'val_precision': test_precision_history,
        'recall': recall_history,
        'val_recall': test_recall_history,
    }

    history(data)


def pred(net, testloader, train=False, show_confusion=False):
    # 用model预测
    total = 0
    correct = 0
    accuracy = 0
    precision = 0
    recall = 0
    count = 0
    with torch.no_grad():
        for data in testloader:
            count += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # print('correct: {0}, total: {1}'.format(predicted.eq(labels.data).sum(), labels.size(0)))
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()
            # if show_confusion:
            labels = labels.cpu()
            predicted = predicted.cpu()
            # roc(labels, predicted)
            acc, pre, rec = performance(['0', '1'], labels, predicted, show_confusion)
            accuracy += acc
            precision += pre
            recall += rec

        if train:
            accuracy_history.append(accuracy/count)
            precision_history.append(precision/count)
            recall_history.append(recall/count)
        else:
            test_accuracy_history.append(accuracy/count)
            test_precision_history.append(precision/count)
            test_recall_history.append(recall/count)

    print('Test\'s ac is: %.3f%%' % (100 * correct / total))

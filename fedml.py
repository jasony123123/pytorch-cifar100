from pydoc import allmethods
from models.resnet import resnet18

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import random
import time


def getCIFAR100TrainMean():
    return [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]


def getCIFAR100TrainSTD():
    return [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]


def getCIFAR100TrainingDataset(dir="./data"):
    data = torchvision.datasets.CIFAR100(
        root=dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(getCIFAR100TrainMean(), getCIFAR100TrainSTD()),
            ]
        ),
    )
    return data


def getCIFAR100TestingDataset(dir="./data"):
    data = torchvision.datasets.CIFAR100(
        root=dir,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(getCIFAR100TrainMean(), getCIFAR100TrainSTD()),
            ]
        ),
    )
    return data


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fn_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def runEntireSimulation(
    trainingDataSplit,
    clientsPerRound,
    iterPerClient,
    batchSize,
    momentum,
    weightDecay,
    epochLRSchedule,
):
    trainDataloaders = [
        torch.utils.data.DataLoader(
            data, batch_size=batchSize, shuffle=True, num_workers=4
        )
        for data in trainingDataSplit
    ]
    testDataloader = torch.utils.data.DataLoader(
        getCIFAR100TestingDataset(), batch_size=batchSize, shuffle=False, num_workers=4
    )

    weights = resnet18().state_dict()
    allMetrics = []

    for (epochs, learningRate) in epochLRSchedule:
        for _ in range(epochs):
            start = time.time()
            weights, trainMetrics = trainFedAvgOneRound(
                weights,
                trainDataloaders,
                clientsPerRound,
                iterPerClient,
                momentum,
                weightDecay,
                learningRate,
            )
            testMetrics = eval(weights, testDataloader)
            allMetrics.append(
                {
                    "train_loss": trainMetrics["loss"],
                    "train_acc1": trainMetrics["acc1"],
                    "test_loss": testMetrics["loss"],
                    "test_acc1": testMetrics["acc1"],
                }
            )
            print(time.time() - start)
            print(allMetrics[-1])
    return weights, allMetrics


def aggregateDict(masterDict, localDict, frac):
    for varName in localDict:
        if varName not in masterDict:
            masterDict[varName] = 0
        masterDict[varName] += localDict[varName] * (1.0 / frac)


def trainFedAvgOneRound(
    weights,
    dataloaders,
    clientsPerRound,
    iterPerClient,
    momentum,
    weightDecay,
    learningRate,
):
    allClientWeights = {}
    allClientMetrics = {}
    whichClients = random.choices(dataloaders, k=clientsPerRound)
    for dataloader in whichClients:
        clientWeights, clientMetrics = trainSgdNRounds(
            weights, dataloader, iterPerClient, momentum, weightDecay, learningRate
        )
        aggregateDict(allClientWeights, clientWeights, clientsPerRound)
        aggregateDict(allClientMetrics, clientMetrics, clientsPerRound)
    return allClientWeights, allClientMetrics


def trainSgdNRounds(
    weights, dataloader, numEpochs, momentum, weightDecay, learningRate
):

    model = resnet18()
    model.load_state_dict(weights)
    model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        model.parameters(), learningRate, momentum=momentum, weight_decay=weightDecay
    )

    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for e in range(numEpochs):
        for i, (input, target) in enumerate(dataloader):
            input, target = input.cuda(), target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec = fn_accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # model.to(torch.device("cpu"))
    weightsToReturn = model.module.state_dict()
    torch.cuda.empty_cache()
    return weightsToReturn, {
        "loss": losses.avg,
        "acc1": top1.avg,
    }


def eval(weights, dataloader):
    criterion = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    top1 = AverageMeter()

    model = resnet18()
    model.load_state_dict(weights)
    model = nn.DataParallel(model).cuda()

    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):
            input, target = input.cuda(), target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec = fn_accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))
    torch.cuda.empty_cache()
    return {"loss": losses.avg, "acc1": top1.avg}

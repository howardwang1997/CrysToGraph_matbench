import time
import os
from multiprocessing.dummy import Pool as ThreadPool
import random
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss, BCEWithLogitsLoss, SmoothL1Loss


class AverageRecorder(object):
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


class Trainer():
    def __init__(self, model, cuda=True):
        self.batch_time = AverageRecorder()
        self.data_time = AverageRecorder()
        self.losses = AverageRecorder()

        self.cuda = cuda and torch.cuda.is_available()
        self.model = model
        if self.cuda:
            self.model = model.cuda()

    def _step(self, inputs):
        if self.cuda:
            inputs = inputs.cuda()
        outputs = self.model(inputs)
        return outputs

    def _calc_loss(self, outputs):
        return self.criterion(outputs[0], outputs[1])

    def train(self, mixed_train_loader, criterion, optimizer, epochs, scheduler=None, verbose_freq: int=100, grad_accum: int=1, classification: bool=False):
        self.model.train()
        self.criterion = criterion
        lrs = True
        if scheduler is None:
            lrs = False
        if self.random_seed is None:
            self.random_seed = [random.random() for i in range(epochs)]
        mpcd = mixed_train_loader.dataset
        batch_size = mixed_train_loader.batch_size
        self.classification = classification

        if self.classification:
            self.criterion_task = BCEWithLogitsLoss()
        else:
            self.criterion_task = MSELoss()

        end = time.time()
        self.loss_list = []
        for epoch in range(epochs):
            loss_list = []
            self.losses.reset()

            for i, data in enumerate(mixed_train_loader):
                self.data_time.update(time.time() - end)
                end = time.time()

                opor = self.model((data[0].to(torch.device('cuda:0')), data[1].to(torch.device('cuda:0'))))
                target = data[-1]

#                target = self._make_target(data[0].y)
                target = Variable(target.float())
                if self.cuda:
                    target = target.cuda()
                loss = self.criterion_task(opor[1], target)

                loss_list.append(loss.data.cpu().item())
                self.losses.update(loss.data.cpu().item(), criterion.batch_size)

                loss /= grad_accum
                loss.backward()

                if i % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                self.batch_time.update(time.time() - end)
                end = time.time()

                if i % verbose_freq == 0:
                    self.verbose(epoch, i, len(mixed_train_loader))
            if lrs:
                scheduler.step()
            self.loss_list.append(loss_list)
            self.save_model()

    def verbose(self, epoch, i, total):
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, i, total, batch_time=self.batch_time,
            data_time=self.data_time, loss=self.losses)
        )

    def save_model(self, path=''):
        if path == '':
            if not hasattr(self, 'save_model_index'):
                names = list(filter(lambda name: 'model' in name, os.listdir('config/')))
                names = [x.split('.')[0] for x in names]
                names = list(filter(lambda name: name[:5] == 'model', names))
                names = list(filter(lambda name: len(name) > 5, names))
                index = max([int(x[5:]) for x in names]) + 1
                self.save_model_index = index
            path = 'config/model%d.tsd' % self.save_model_index
        torch.save(self.model.state_dict(), path)


    def load_model(self, path=''):
        if path == '':
            path = 'config/model.tch'
        return torch.load(path)

    def save_state_dict(self, path=''):
        if path == '':
            path = 'config/model.tsd'
        torch.save(self.model.state_dict(), path)


# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            conv1=L.Convolution2D(3, 96, 11, stride=6),
            bn2=L.BatchNormalization(96),
            conv3=L.Convolution2D(96, 256, 5, stride=3),
            bn4=L.BatchNormalization(256),
            conv5=L.Convolution2D(256, 384, 3, pad=1),
            conv6=L.Convolution2D(384, 384, 3, pad=1),
            conv7=L.Convolution2D(384, 256, 3, pad=1),
            bn8=L.BatchNormalization(256),
            fc9=L.Linear(1024, 4096),
            fc10=L.Linear(4096, 4096),
            fc11=L.Linear(4096, 25),
        )

    def __call__(self, x, train=True):
        h = F.max_pooling_2d(self.bn2(F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(self.bn4(F.relu(self.conv3(h))), 2, stride=2)
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pooling_2d(self.bn8(F.relu(self.conv7(h))), 2, stride=2)
        h = F.dropout(F.relu(self.fc9(h)), train=train)
        h = F.dropout(F.relu(self.fc10(h)), train=train)
        y = self.fc11(h)

        return y

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)
        self.train = True

    def __call__(self, x, t, train=True):
        y = self.predictor(x, train)
        self.loss = F.softmax_cross_entropy(y, t)
        self.acc = F.accuracy(y, t)

        return self.loss

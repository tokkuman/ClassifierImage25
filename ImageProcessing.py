# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import csv
import sys
import time
import random
import copy
import math
import os
import os.path as pt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as skimage
from skimage import transform as tr
from argparse import ArgumentParser

class ImageData():
    def __init__(self, img_dir, meta_data):
        self.img_dir = img_dir
        assert meta_data.endswith('.tsv')
        self.meta_data = pd.read_csv(meta_data, sep='\t')
        self.index = np.array(self.meta_data.index)
        self.split = False

    # Split Train and Evaluation Data
    def splitData(self, train_size):
        self.trainIndex = np.random.choice(self.index, train_size, replace=False)
        self.evalIndex = np.array([i for i in self.index if i not in self.trainIndex])
        self.split = True
        
    # Shuffle of Training Data
    def shuffleData(self):
        assert self.split == True
        self.trainIndex = np.random.permutation(self.trainIndex)

    # Make Minibatch of Read Image and Resize
    def makeMinibatch(self, batchsize, imSize=256, mode=None):
        if mode == 'train':
            assert self.split == True
            meta_data = self.meta_data.ix[self.trainIndex]
            idx = self.trainIndex

        elif mode == 'eval':
            assert self.split == True
            meta_data = self.meta_data.ix[self.evalIndex]
            idx = self.evalIndex

        else:
            meta_data = self.meta_data
            idx = self.index

        i = 0
        while i < len(idx):
            data = meta_data.iloc[i:i+batchsize]
            images = []
            for f in list(data['file_name']):
                img = skimage.imread(os.path.join(self.img_dir, f))
                ipsize = (imSize, imSize, 3)
                ipImg = tr.resize(img, ipsize)
                images.append(np.array(ipImg))
            images = np.array(images)
            images = images.transpose((0,3,1,2))
            images = images.astype(np.float32)

            if 'category_id' in data.columns:
                labels = np.array(list(data['category_id']))
                labels = labels.astype(np.int32)
                yield images, labels
            else:
                yield images
            i += batchsize

# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.links import caffe
from chainer import link

import csv
import sys
import time
import random
import copy
import math
import six
import os
import os.path as pt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as skimage
from skimage import transform as tr
from argparse import ArgumentParser
import zipfile

from model import Model
from model import Classifier
from ImageProcessing import ImageData


ap = ArgumentParser(description='python main.py')
ap.add_argument('--inputdir', '-i', nargs='?', default='trainData', help='Specify input files directory training data')
ap.add_argument('--inputgt', '-g1', nargs='?', default='clf_train_master.tsv', help='Specify input answer files')
ap.add_argument('--ansdir', '-a', nargs='?', default='testData', help='Specify input test files directory')
ap.add_argument('--ansgt', '-g2', nargs='?', default='clf_test.tsv', help='Specify input test files')
ap.add_argument('--outdir', '-o', nargs='?', default='Result', help='Specify output files directory for create result and save model file')
ap.add_argument('--model', '-m', nargs='?', default='0', help='Specify loading file path of learned Model')
ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
ap.add_argument('--trainsize', '-s', type=int, default=9000, help='Specify size of training data')
ap.add_argument('--epoch', '-e', type=int, default=10, help='Specify number of sweeps over the dataset to train')
ap.add_argument('--batchsize', '-b', type=int, default=10, help='Specify batchsize')

args = ap.parse_args()
opbase = args.outdir
argvs = sys.argv

# GPU use flag
print 'GPU: {}'.format(args.gpu)
# Path Separator
psep = '/'
# OutDir の端に file sepaeator があれば削除    
if (args.outdir[len(opbase) - 1] == psep):
    opbase = opbase[:len(opbase) - 1]
# OutDir が絶対 Path で無い場合は Current Directory を追加
if not (args.outdir[0] == psep):
    if (args.outdir.find('./') == -1):
        opbase = './' + opbase
# Create Opbase
t = time.ctime().split(' ')
if t.count('') == 1:
    t.pop(t.index(''))
opbase = opbase + '_' + t[1] + t[2] + t[0] + '_' + t[4] + '_' + t[3].split(':')[0] + t[3].split(':')[1] + t[3].split(':')[2]
# Output Directory の有無を Check
if not (pt.exists(opbase)):
    os.mkdir(opbase)
    print 'Output Directory not exist! Create...'
print 'Output Directory:', opbase


def TrainingFhase(trainData, clf, opt):
    # split train and eval data
    trainData.splitData(args.trainsize)
    for epoch in range(args.epoch):
        ### Training
        numSamples = 0
        trainSumLoss = 0
        trainSumAcc = 0
        for data in trainData.makeMinibatch(args.batchsize, mode='train'):
            numSamples += len(data[0])
            x, y = Variable(data[0]), Variable(data[1])
            if args.gpu == 0:
                x.to_gpu()
                y.to_gpu()
            loss = clf(x, y, train=True)
            opt.zero_grads()
            loss.backward()    # back propagation
            opt.update()       # update parameters
            trainSumAcc += clf.acc.data * args.batchsize
            trainSumLoss += clf.loss.data * args.batchsize
        trainAcc = trainSumAcc / numSamples
        trainLoss = trainSumLoss / numSamples

        ### Evaluation
        numSamples = 0
        evalSumLoss = 0
        evalSumAcc = 0
        for data in trainData.makeMinibatch(args.batchsize, mode='eval'):
            numSamples += len(data[0])
            x, y = Variable(data[0]), Variable(data[1])
            if args.gpu == 0:
                x.to_gpu()
                y.to_gpu()
            loss = clf(x, y, train=False)
            evalSumAcc += clf.acc.data * args.batchsize
            evalSumLoss += clf.loss.data * args.batchsize
        evalAcc = evalSumAcc / numSamples
        evalLoss = evalSumLoss / numSamples

        print '===================================='
        print 'epoch :', epoch+1
        print 'TrainLoss :', trainLoss, ', TrainAccuracy :', trainAcc
        print 'TestLoss :', evalLoss, ', TestAccuracy :', evalAcc 

        filename = opbase + psep + 'result.txt'
        f = open(filename, 'a')
        f.write('==================================\n')
        f.write('epoch : '.format(str(epoch+1)) + '\n')
        f.write('TrainLoss={}, TrainAccuracy={}'.format(trainLoss, trainAcc) + '\n')
        f.write('TestLoss={}, TestAccuracy={}'.format(evalLoss, evalAcc) + '\n')
        f.close()

        # Save Model
        modelfile = 'Modelhdf5_' + str(epoch+1) + '.model'
        serializers.save_hdf5(opbase + psep + modelfile, model)
        modelfile = 'Modelnpz_' + str(epoch+1) + '.model'
        serializers.save_npz(opbase + psep + modelfile, model)

        # Shuffle of Train Data
        trainData.shuffleData()

    return clf, opt


def predictPhase(testData, clf):
    if args.gpu == 0:
        clf.predictor.to_gpu()
    else:
        clf.predictor.to_cpu()
    numSamples = 0
    preResult = np.zeros((len(testData.index),25))
    for data in testData.makeMinibatch(args.batchsize):
        num_samples += len(data)
        x = Variable(data)
        if gpu == 0:
            x.to_gpu()
        ypre = clf.predictor(x, train=False)
        ypre = F.softmax(ypre)
        ypre.to_cpu()
        preResult[numSamples-len(data):numSamples,:] = ypre.data
    return preResult

if __name__ == '__main__':
    model = Model()
    if not args.model == '0':
        try:
            serializers.load_hdf5(args.model, model)
            print 'Loading Model : ' + args.model
            filename = opbase + psep + 'result.txt'
            f = open(filename, 'w')
            f.write('Loading Model : {}\n'.format(args.model))
            f.close()
        except:
            print 'ERROR!!'
            print 'Usage : Input File Path of Model (ex ./hoge.model)'
            sys.exit()
    if args.gpu == 0:
        cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()
    clf = Classifier(model)
    opt = optimizers.Adam()
    opt.setup(clf)
    opt.add_hook(chainer.optimizer.WeightDecay(0.0001))

    filename = opbase + psep + 'result.txt'
    f = open(filename, 'w')
    f.write('python ' + ' '.join(argvs) + '\n')
    f.write('[Hyperparameter of Learning Properties]\n')
    f.write('Output Directory : {}\n'.format(opbase))
    f.write('GPU: {}\n'.format(args.gpu))
    f.write('number of Training Data : {}\n'.format(args.trainsize))
    f.close()

    trainData = ImageData(args.inputdir, args.inputgt)  # Read of Training Data
    testData = ImageData(args.ansdir, args.ansgt)       # Read of Evaluation Data

    clf, opt = TrainingFhase(trainData, clf, opt)       # Training & Validatioin
    pre = predictPhase(testData, clf)                   # Evaluation


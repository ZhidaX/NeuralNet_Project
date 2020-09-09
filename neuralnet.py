# This is a single-hidden-layer Neural Network implemented as part of a machine learning class
# that I took. It is built completely from the ground up, with negligible reliance on libraries
# and external packages. -Andy Xu


import numpy as np
import math
import sys
import csv

#examples of commands that run this script. Inputs from the command line are required(see main).
#python neuralnet.py smallTrain.csv smallTest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 2 0.1
#python neuralnet.py tinyTrain.csv tinyTest.csv tinyTrain_out.labels tinyTest_out.labels tinyMetrics_out.txt 1 4 2 0.1



class neuralnet(object):
    def __init__(self):
        self.alpha = np.random.uniform(-0.1, 0.1, (hiddenUnits, trainData.shape[1])) if initFlag == 1 else np.zeros((hiddenUnits, trainData.shape[1]))
        self.a = None
        self.beta = np.random.uniform(-0.1, 0.1, (10, hiddenUnits+1)) if initFlag == 1 else np.zeros((10, hiddenUnits+1))
        self.z = None
        self.b = None
        self.yhat = None

def activate(z):
    return np.exp(z)/(1+np.exp(z))

def forward(x, brain):
    brain.a = np.dot(brain.alpha, x)
    brain.z = np.insert(activate(brain.a), 0, 1)
    brain.b = np.dot(brain.beta, brain.z)
    soft = np.exp(brain.b)
    brain.yhat = soft / np.sum(soft)
    return brain

def backward(brain, x, y):
    yhat_ = np.copy(brain.yhat)
    yhat_[y] -= 1
    b_ = np.expand_dims(yhat_, axis=1)
    z = np.expand_dims(brain.z, axis=1)
    beta_ = np.dot(b_, z.T)
    betatrunc = np.delete(brain.beta, 0, 1)
    z_ = np.dot(b_.T, betatrunc).T
    zmod = np.delete(z,0,0)
    a_ = np.multiply(np.multiply(zmod,1-zmod),z_)
    alpha_ = np.dot(a_, np.expand_dims(x, axis=0))
    return alpha_, beta_

def entropy(brain, data):
    ent = 0
    for datum in data:
        y = datum[0]
        x = datum[1:]
        if x.shape != (trainData.shape[1],): x = np.insert(x, 0, 1)
        brain = forward(x, brain)
        ent += math.log(brain.yhat[y])
    return -ent / max(np.shape(data)[0], 1)

def SGD():#Stands for Stochastic Gradient Descent
    brain = neuralnet()
    for _ in range(numEpoch):
        for datum in trainData:
            y = datum[0]
            x = datum[1:]
            if x.shape != (trainData.shape[1],): x = np.insert(x, 0, 1)
            brain = forward(x, brain)
            dAlpha, dBeta = backward(brain, x, y)
            brain.alpha -= learningRate*dAlpha
            brain.beta -= learningRate*dBeta
        metricsOut.write("epoch=%d crossentropy(train): %f" %(_ + 1, entropy(brain, trainData)) + "\n")
        metricsOut.write("epoch=%d crossentropy(test): %f" %(_ + 1, entropy(brain, testData)) + "\n")
    return brain


if __name__ == '__main__':

    # Acquire input

    trainInput = csv.reader(open(sys.argv[1]), delimiter=',')
    testInput = csv.reader(open(sys.argv[2]), delimiter=',')
    trainOut = open(sys.argv[3], "w+")
    testOut = open(sys.argv[4], "w+")
    metricsOut = open(sys.argv[5], "w+")
    numEpoch = int(sys.argv[6])
    hiddenUnits = int(sys.argv[7])
    initFlag = int(sys.argv[8])
    learningRate = float(sys.argv[9])


    # The main function begins


    trainData = []
    testData = []
    for row in trainInput:
        trainData.append(row)
    trainData = np.array(trainData,dtype=int)
    for row in testInput:
        testData.append(row)
    testData = np.array(testData,dtype=int)

    # setup ends
    
    brain = SGD()
    trainerror = 0
    testerror = 0
    for datum in trainData:
        y = datum[0]
        x = datum[1:]
        if x.shape != (trainData.shape[1],): x = np.insert(x,0,1)
        brain = forward(x, brain)
        pred = np.argmax(brain.yhat)
        trainOut.write(str(pred) + "\n")
        if pred != y: trainerror += 1
    for datum in testData:
        y = datum[0]
        x = datum
        if x.shape != (trainData.shape[1],): x = np.insert(x,0,1)
        brain = forward(x, brain)
        pred = np.argmax(brain.yhat)
        testOut.write(str(pred) + "\n")
        if pred != y: testerror += 1
    
    # the stats are out

    metricsOut.write("error(train): " + str(trainerror/trainData.shape[0]) + "\n")
    metricsOut.write("error(test): " + str(testerror/testData.shape[0]))
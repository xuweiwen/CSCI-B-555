from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import collections
import classalgorithms as algs
 
def splitdataset(dataset, trainsize=10000, testsize=5000, testfile=None):
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    Xtrain = dataset[randindices[0:trainsize],0:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],0:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]
    if testfile is not None:
        testdataset = loadcsv(testfile)
        Xtest = dataset[:,0:numinputs]
        ytest = dataset[:,numinputs]        
        
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))
 
 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def loadsusy():
    dataset = np.genfromtxt('susyall.csv', delimiter=',')
    trainset, testset = splitdataset(dataset)    
    return trainset,testset

def loadmadelon():
    datasettrain = np.genfromtxt('madelon/madelon_train_norm.data', delimiter=' ')
    trainlab = np.genfromtxt('madelon/madelon_train.labels', delimiter=' ')
    trainlab[trainlab==-1] = 0
    trainsetx = np.hstack((datasettrain, np.ones((datasettrain.shape[0],1))))
    trainset = (trainsetx, trainlab)
    
    datasettest = np.genfromtxt('madelon/madelon_valid_norm.data', delimiter=' ')
    testlab = np.genfromtxt('madelon/madelon_valid.labels', delimiter=' ')
    testlab[testlab==-1] = 0
    testsetx = np.hstack((datasettest, np.ones((datasettest.shape[0],1))))
    testset = (testsetx, testlab)
      
    return trainset,testset

if __name__ == '__main__':
    """You have to comment out the line containing loadsusy() or loadmadelon() depending on which dataset you want to run the algorithms on"""
    trainset, testset = loadsusy()
    obj=[]
    #trainset, testset = loadmadelon()
    print('Running on train={0} and test={1} samples').format(trainset[0].shape[0], testset[0].shape[0])
    nnparams = {'ni': trainset[0].shape[1], 'nh': 64, 'no': 1}
    """type parameter should be L1,L2,None or Other"""
    """regwt should be user defined parameter"""
    
    lrparms={'regwt':0,'type':"None"}
    classalgs = {'Random': algs.Classifier(),
                 'Linear Regression': algs.LinearRegressionClass(),
                 'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 'Naive Bayes Ones': algs.NaiveBayes(),
                 'My Classifier': algs.MyClassifier(),
                 'Logistic Regression': algs.LogitReg(lrparms),
                 'Neural Network': algs.NeuralNet(nnparams)
                 }

    classalgs1 = collections.OrderedDict(sorted(classalgs.items()))
                          
    for learnername , learner in classalgs1.iteritems():
        print 'Running learner = ' + learnername
        
        # Train model
        if learnername=="Linear Regression":
           lobj=learner  
        
        if learnername=="Logistic Regression":
           learner.learn(trainset[0], trainset[1],lobj)
        else:
           learner.learn(trainset[0], trainset[1],None) 
        
        # Test model   
        predictions = learner.predict(testset[0])
        accuracy = getaccuracy(testset[1], predictions)
        print 'Accuracy for ' + learnername + ': ' + str(accuracy)
 

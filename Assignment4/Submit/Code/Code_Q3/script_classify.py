from __future__ import division # floating point division
import csv
import sys
import random
import numpy as np
#import matplotlib.pyplot as plt
import algorithms as algs
import utilities as utils

######You need to enter the parameter "split" with the code to run it on a
#training set of the first 20000 samples and testset of the next 5000 samples.

def splitdataset(dataset, trainsize=20000,testsize=10000):
    # Now randomly split into train and test    

    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    offset = 50 # Ignore the first 50 features
    Xtrain = dataset[randindices[0:trainsize],offset:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],offset:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]
    
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))
    
def geterror(predictions, ytest):
    # Can change this to other error values
    print("testset",ytest.shape[0])
    #print 200000/ytest.shape[0]
    return utils.l2err_squared(predictions,ytest)/ytest.shape[0]
########The Main Class execution starts from here########################
    
if __name__ == '__main__':
    acclist=[]
    #filename = 'blogData_train.csv'
    
    filename='blogData_train.csv'
    dataset = utils.loadcsv(filename)
    trainset,testset = splitdataset(dataset)
    #  ntrainset,ntestset=splitdataset(dataset1)
    print('Split {0} rows into train={1} and test={2} rows').format(
    len(dataset), trainset[0].shape[0], testset[0].shape[0])
    klparm={'regwt':0.1,'sigma':0,'method':"linear",'const':0.01}
    classalgs = {
                 'Random': algs.Regressor(),
                 'Mean': algs.MeanPredictor(),
                 'RidgeRegression': algs.RidgeRegression({'regwt':0.1}),
                 'FSLinearRegression': algs.FSLinearRegression(trainset[0]),
                 'Kernel Regression':algs.KernelRegression(klparm)
                 }

    # Runs all the algorithms on the data and print out results    
    for learnername, learner in classalgs.iteritems():
        print 'Running learner = ' + learnername
        
        # Train model
        if learnername=='Kernel Regression':
           learner.clustering(trainset[0])
           learner.build_new_feature(trainset[0],"train")
          
        learner.learn(trainset[0], trainset[1])
        
        # Test model
        if learnername=="Kernel Regression":
           learner.build_new_feature(testset[0],"test") 
           
        predictions = learner.predict(testset[0])
        
        #print predictions
        accuracy = geterror(testset[1], predictions)
        print 'Accuracy for ' + learnername + ': ' + str(accuracy)
              
              
           
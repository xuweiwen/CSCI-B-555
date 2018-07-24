from __future__ import division  # floating point division\
#from copy import deepcopy
from random import shuffle
import csv
import random
import math
#import sys
import numpy as np
import collections

import classalgorithms as algs
 
def splitdataset(dataset, trainsize=20000, testsize=10000, testfile=None):
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

"""10-fold cross validation fold creation"""    
def cross_validate_dataset(slices,fold,numinputs,count):
            
    validation = slices[count]
    training = [item
                for s in slices if s is not validation
                for item in s]
    
    validation=np.asarray(validation)
    training=np.asarray(training) 
               
    Xtrain = training[:,0:numinputs]
    ytrain = training[:,numinputs]
    Xtest =  validation[:,0:numinputs]
    ytest =  validation[:,numinputs]
    
    return ((Xtrain,ytrain),(Xtest,ytest))
    
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
    #sys.stdout = open("D:\Grad Studies\Machine Learning\Assignment4\Question3\log.txt", "w")
    
    final_accuracy={"Linear Regression":[],"Naive Bayes Ones":[],
                    "Logistic Regression":[],"Neural Network":[]} 
                    
    parm_dict={ 'Linear Regression':[0],
                'Naive Bayes Ones':[0],
                'Logistic Regression':[0.0001,0.001,0.01,0.1,0,1,10,100,1000,10000],
                'Neural Network':[40,48,64,80,96,112,128,144,160,176,192]}
                    
    for repeat in xrange(40):
        
       trainset, testset = loadsusy()
       """The choice of the number of folds should be user-input"""
       fold=10
    
       trainlabel=np.reshape(trainset[1],(-1,1))
       trset = np.hstack((trainset[0],trainlabel))
       numinputs = trset.shape[1]-1
       np.random.shuffle(trset)
       parts = [trset[i::fold] for i in xrange(fold)]
       obj=[] 
       print('Running on train={0} and test={1} samples').format(trainset[0].shape[0], testset[0].shape[0])
       parm_pass={'Neural Network':{'ni': trset.shape[1]-1, 'nh': 0, 'no': 1},
               'Logistic Regression':{'regwt':0,'type':"L2"}}
               
       classalgs = {'Linear Regression': algs.LinearRegressionClass(),
                    'Naive Bayes Ones': algs.NaiveBayes(),
                    'Logistic Regression': algs.LogitReg(parm_pass['Logistic Regression']),
                    'Neural Network': algs.NeuralNet(parm_pass['Neural Network'])
                 }
                 
       classalgs1 = collections.OrderedDict(sorted(classalgs.items())) 
        
       best_parm=[]
       
       for learnername , learner in classalgs1.iteritems():
        
           print 'Running learner = ' + learnername
        
#           # Train model
           parm_accuracy={}
        
           for j in range(0,len(parm_dict[learnername])):
               parm=[]
               algo_accuracy=[]
               for i in xrange(fold):
            
                   trainset1,validation = cross_validate_dataset(parts,fold,numinputs,i)
            
                   if learnername=="Linear Regression":
                      lobj=learner  
               
                   if learnername=="Logistic Regression":
               
                      parm_pass['Logistic Regression']['regwt']=parm_dict[learnername][j]
                      print("Running Logistic regression with regularisation parm",parm_dict[learnername][j])
                      learner=algs.LogitReg(parm_pass['Logistic Regression'])
                      learner.learn(trainset1[0], trainset1[1],lobj)
            
                   elif learnername=="Neural Network":
                    
                        parm_pass['Neural Network']['nh']=parm_dict[learnername][j]
                        print("Running Neural Network with hidden number of nodes",parm_dict[learnername][j])
                        learner=algs.NeuralNet(parm_pass['Neural Network'])
                        learner.learn(trainset1[0], trainset1[1],lobj)
                   
                   else:       
                        learner.learn(trainset1[0], trainset1[1],None) 
#        
#                  # Test model   
                   predictions = learner.predict(validation[0])
             
                   """Calculating the Accuracy"""
                   accuracy = getaccuracy(validation[1], predictions)
                   algo_accuracy.append(accuracy)
             
               print("The each fold accuracy of "+ learnername +"  ")
               print(algo_accuracy)            
               avg=sum(algo_accuracy)/fold
               print 'Average accuracy for ' + learnername + ': ' + str(avg)


               if learnername=="Linear Regression" or learnername=="Naive Bayes Ones": 
                  break 
               else:
                  parm_accuracy[parm_dict[learnername][j]]=avg
                              
               """Choosing the best parameter"""
               print (parm_accuracy)
        
           if learnername=="Logistic Regression" or learnername=="Neural Network":        
              maxitem=0
              bestparm=0
              for parm , parm_accur in parm_accuracy.iteritems():   
                  if parm_accur>maxitem:
                     maxitem=parm_accur
                     bestparm=parm
              best_parm.append(bestparm)       
              print("The best parameter for"+learnername+"is:  "+str(bestparm)+"with accuracy  "+ str(maxitem))       
              print (best_parm)
           
       parm_pass['Logistic Regression']['regwt']=best_parm[0]
       parm_pass['Neural Network']['nh']=best_parm[1]      
       print parm_pass

       """Run the model on Training data with the best parameter"""
    
       for learnername , learner in classalgs1.iteritems():
           print 'Running learner = ' + learnername
        
           if learnername=="Linear Regression":
              lobj=learner
                   
           if learnername=="Logistic Regression":
              learner= algs.LogitReg(parm_pass['Logistic Regression'])
              learner.learn(trainset[0], trainset[1],lobj)
           elif learnername=="Neural Network":
                learner=algs.NeuralNet(parm_pass['Neural Network'])
                learner.learn(trainset[0], trainset[1],None) 
           else:
               learner.learn(trainset[0], trainset[1],None)
           
           """Run the model on the new test data"""   
           predictions = learner.predict(testset[0])
           accuracy = getaccuracy(testset[1], predictions)
           print 'Accuracy for ' + learnername + ': ' + str(accuracy)
           final_accuracy[learnername].append(accuracy)
       print("-----------------------------------------------------------------------")
       print("-----------------------------------------------------------------------")
        
    print (final_accuracy)   
    temp=[]
    for  learnername , accr_queue in final_accuracy.iteritems():
         for i in range(0,len(accr_queue)):
             temp.append(accr_queue[i])
    print("The length of temp is",len(temp))         
    for j in range(0,len(temp)):
        """error.csv is the file tat captures errors for all the algorithms across 40 runs"""
        with open('error.csv','ab') as f1:
            csv_writer = csv.writer(f1,delimiter=',',dialect = 'excel')
            row=(temp[j],"\n")
            print(row)
            csv_writer.writerows([row])

        
            
 

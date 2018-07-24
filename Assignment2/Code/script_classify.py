from __future__ import division # floating point division
import csv
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import algorithms as algs
import utilities as utils

######You need to enter the parameter "split" with the code to run it on a
#training set of the first 20000 samples and testset of the next 5000 samples.

def splitdataset1(dataset, trainsize=20000,testsize=5000):
    # Now randomly split into train and test    

    #randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    offset = 50 # Ignore the first 50 features
    Xtrain = dataset[0:trainsize,offset:numinputs]
    ytrain = dataset[0:trainsize,numinputs]
    t1=trainsize+1000
    Xtest = dataset[t1:t1+testsize,offset:numinputs]
    ytest = dataset[t1:t1+testsize,numinputs]
    
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
    
    return ((Xtrain,ytrain ), (Xtest,ytest))

 
def splitdataset(dataset, trainsize=30000,testsize=10000):
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
    
def create_test_train(trainset,testset,parm,fold):
    
    trainset1=[]
    testset1=[]
    
    for i in range(0,len(trainset)):
        trainset1.append(trainset[i])
        testset1.append(testset[i])
    
    trainset1.pop(parm)
    testset1.pop(parm)
    Xtrain=np.vstack(trainset1)
    ytrain=np.hstack(testset1)
                    
    Xtest=trainset[parm]
    ytest=testset[parm]
    
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
        
    return ((Xtrain,ytrain), (Xtest,ytest))
    
def cross_validate_dataset(dataset,fold):
    
    # Now randomly split into train and test
    
    eachfold=dataset.shape[0]/fold
    offset = 50
    numinputs = dataset.shape[1]-1
    Xtrain=[]
    Ytrain=[]
    start=0
    end=eachfold
    
    for i in range(0,fold):
        Xtrain.append(dataset[start:end,offset:numinputs])
        Ytrain.append(dataset[start:end,numinputs])
        start=end
        end=end+eachfold
   
    return ((Xtrain,Ytrain))
 
 
def geterror(predictions, ytest):
    # Can change this to other error values
    print("testset",ytest.shape[0])
    print 200000/ytest.shape[0]
    return utils.l2err_squared(predictions,ytest)/ytest.shape[0]
########The Main Class execution starts from here########################
    
if __name__ == '__main__':
    acclist=[]
    #filename = 'blogData_train.csv'
    
    filename='blogData_train_norm.csv'
    dataset = utils.loadcsv(filename)

    if sys.argv[1]=="split":
       trainset,testset = splitdataset1(dataset)
     #  ntrainset,ntestset=splitdataset(dataset1)
       print('Split {0} rows into train={1} and test={2} rows').format(
       len(dataset), trainset[0].shape[0], testset[0].shape[0])
       classalgs = {
                 # 'Random': algs.Regressor(),
                 # 'Mean': algs.MeanPredictor(),
                 # 'RidgeRegression': algs.RidgeRegression(),
                 # 'FSLinearRegression': algs.FSLinearRegression(trainset[0]),
                  #'Stochastic Gradient Descent': algs.StochasticGrad
                  #  (trainset[0])
                  'Poisson Regression':algs.PRegression(trainset[0])
                 }

    # Runs all the algorithms on the data and print out results    
       for learnername, learner in classalgs.iteritems():
           print 'Running learner = ' + learnername
        # Train model
           
           if learnername=="RidgeRegression":
              alpha=0.1
              #learner.feature_select(trainset[0],trainset[1]) 
              learner.learn(trainset[0], trainset[1],alpha)
              # Test model
              predictions = learner.predict(testset[0])
              #print predictions
              accuracy = geterror(testset[1], predictions)
              print 'The alpha is'+ str(alpha)
              print 'Accuracy for ' + learnername + ': ' + str(accuracy)
              alpha=alpha+1
              
           elif learnername =="Stochastic Gradient Descent":
                lrate=0.00001
                flag=True
                alpha=1
                for i in range(0,trainset[0].shape[0]):
                   flag=learner.learn(trainset[0][i:i+1,],trainset[1][i],lrate,alpha)
                   #print i
                   if flag== False:
                       break
                 #   learner.bounding_weight()
                 #   if i==1:
                 #     break
                # Test model
                predictions = learner.predict(testset[0])
                #print predictions
                accuracy = geterror(testset[1], predictions)
                print 'The alpha is'+ str(alpha)
                print 'Accuracy for ' + learnername + ': ' + str(accuracy)
                
           elif learnername =="Poisson Regression":
                #learner.feature_select(trainset[0],trainset[1]) 
                iterflag=True
                maxtest1=np.amax(trainset[1])
                print maxtest1                
                for i in range(0,len(trainset[1])):
                    trainset[1][i]=trainset[1][i]/maxtest1
                    
                for i in range(0,len(testset[1])):
                    testset[1][i]=testset[1][i]/maxtest1
                    
                print testset[1]
                    
                while(iterflag):
                    alpha=0.001
                    flag=learner.learn(trainset[0],trainset[1],alpha)
                    iterflag=flag
                    # Test model
                predictions = learner.predict(testset[0])
                #print predictions
                testset[1]=np.multiply(testset[1],1/np.amax(testset[1]))
                accuracy = geterror(testset[1], predictions)
                print 'The alpha is'+ str(alpha)
                print 'Accuracy for ' + learnername + ': ' + str(accuracy)
           else:     
                #learner.feature_select(trainset[0],trainset[1])            
                learner.learn(trainset[0],trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                #print predictions
                accuracy = geterror(testset[1], predictions)
                print 'Accuracy for ' + learnername + ': ' + str(accuracy)
           
    elif sys.argv[1]=="cross":
 #############################################################################       
        #This is an user entered parameter
            fold=10
        #---------------------------------------------------------------------    
            parm=0
            ctrainset,ctestset= cross_validate_dataset(dataset,fold)
            first=True
            for i in range(0,fold):
              #parm=0  
              trainset=None
              testset=None
              trainset,testset=create_test_train(ctrainset,ctestset,parm,fold)
              print('Split {0} rows into train={1} and test={2} rows').format(
              len(dataset), trainset[0].shape[0], testset[0].shape[0])
              classalgs ={'RidgeRegression': algs.RidgeRegression()
                          #'Mean': algs.MeanPredictor(),
                          #'Random': algs.Regressor()
                         # 'Stochastic Gradient Descent': algs.StochasticGrad(trainset[0])
                          }
              # {'Random': algs.Regressor(),
              # 'Mean': algs.MeanPredictor(),
              # 'RidgeRegression': algs.RidgeRegression()
              # 'FSLinearRegression': algs.FSLinearRegression()
              #          }

    # Runs all the algorithms on the data and print out results    
              for learnername, learner in classalgs.iteritems():
                  print 'Running learner = ' + learnername
        # Train model
                  if learnername=="RidgeRegression":
                     alpha=1
                     while(alpha<=1):
                         if (first):
                            learner.feature_select(trainset[0],trainset[1])
                            first=False
                         learner.learn(trainset[0], trainset[1],alpha)
            
                         # Test model
                         predictions = learner.predict(testset[0])
                         #print predictions
                         
                         accuracy = geterror(testset[1], predictions)
                         acclist.append(accuracy)
                         print 'The alpha is'+ str(alpha)
                         print 'Accuracy for ' + learnername + ': ' + str(accuracy)
                         print("The fold is"+str(parm))
                         alpha=alpha+0.1
                         
                  else:
                      learner.learn(trainset[0], trainset[1])
                      # Test model
                      predictions = learner.predict(testset[0])
                      #print predictions
                      accuracy = geterror(testset[1], predictions)
                      print 'Accuracy for ' + learnername + ': ' + str(accuracy)
 
              parm=parm+1
              
            avgacrcy=0

            for i in range(0,fold):
                avgacrcy=avgacrcy+acclist[i]
                
            avgacrcy=avgacrcy/fold
            print("The average accuracy across 10-fold is",avgacrcy)
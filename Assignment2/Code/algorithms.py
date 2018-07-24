from __future__ import division # floating point division
import numpy as np
import utilities as utils
import matplotlib.pyplot as plt
from random import randint
import math

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects value between max and min in training set.
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, params=None ):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self,Xtrain):
        self.weights = None
        self.features = [1,2,3,4,5,6,7,8,9,10]
        
        #for i in range (1,Xtrain.shape[1]):
        #   self.features.append(i)
            
        #print(self.features)
    def feature_select(self,Xtrain,ytrain):
        
        #The elements in hcor are the indexes of the highly correlated features.
        #Col 228 contains all zeros, so it was removed else correlation was not getting calculated properly.
        ##################################################################
        # This R code was used to remove correlated features
        #Reference http://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
        #library(mlbench)
        #library(caret)
        #data<-read.csv("dataBlog_train_norm.csv",header=FALSE)
        #data1<-data[,51:277]
        #data2<-data[,279:280]
        #dataf<-cbind(data1,data2)
        #cmat <- cor(data)
        #hcor <- findCorrelation(cmat, cutoff=0.75)
        #print hcor
        ######################################################################
        
        Xtrain1=np.copy(Xtrain)
        Xtrain1=np.delete(Xtrain1,-1,1)
        #hcor=[4,44,47,80,82,99,106,111,116,119,122,129,140,148,150,162,
        #      186,193,196,200,1,6,15,23,26,30,31,228,229]
        hcor=[3,43,46,79,81,98,105,110,115,118,121,128,139,147,149,161,
              185,192,195,199,0,5,14,22,25,29,30,227,228]
        alfeat=[]
        print("The shape is",Xtrain1.shape)
        for i in range(0,Xtrain1.shape[1]):
            alfeat.append(i)
            #print(alfeat[i])
        for i in range(0,10):
            while(True):
              a=randint(0,229)
              if a in hcor:
                 continue
              else:
                  break
            self.features.append(a)
            
        print self.features
                  
            
            
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:,self.features]
        Xless = np.hstack((Xless, np.ones((Xless.shape[0],1))))
        print ["The shape of Xless is",Xless.shape]
        #print Xtrain.shape[1]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]     
        Xless = np.hstack((Xless, np.ones((Xless.shape[0],1))))
        ytest = np.dot(Xless, self.weights) 
        #print("I am plotting")
        #plt.plot(Xtrain, ytrain)
        return ytest
        
class RidgeRegression(Regressor):
    """
    Ridge Regression 
    """
    def __init__( self, params=None ):
        self.weights = None
        self.features = []  
        
    def feature_select(self,Xtrain,ytrain):
        
        Xtrain1=np.copy(Xtrain)
        Xtrain1=np.delete(Xtrain1,-1,1)
        hcor=[3,43,46,79,81,98,105,110,115,118,121,128,139,147,149,161,
              185,192,195,199,0,5,14,22,25,29,30,227,228]
        alfeat=[]
        print("The shape is",Xtrain1.shape)
        for i in range(0,Xtrain1.shape[1]):
            alfeat.append(i)
            #print(alfeat[i])
        for i in range(0,10):
            while(True):
              a=randint(0,229)
              if a in hcor:
                 continue
              else:
                  break
            self.features.append(a)
            
        print self.features
    
#take all the features in the dataset    
        #for i in range (1,11):
        #    self.features.append(i)
        
    def learn(self, Xtrain, ytrain,alpha):
        """ Learns using the traindata """
        """ The parameter alpha is user defined. When alpha=0, then the solution is OLS regrssion """
        
        print ("Xtrain dim is",Xtrain.shape)
        #Xless = Xtrain[:,self.features]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)+ np.multiply(alpha,np.identity(Xtrain.shape[1]))), Xtrain.T),ytrain)
        #self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)+ 
        #np.multiply(alpha,np.identity(Xless.shape[1]))), Xless.T),ytrain)
        #print self.weights
        
    def predict(self, Xtest):
        #Xless = Xtest[:,self.features]   
        #print ("Xtest dim is",Xtest.shape)     
        #ytest = np.dot(Xtest, self.weights)  
        ytest = np.dot(Xtest, self.weights)
        
        return ytest
        
class StochasticGrad():
    
#Code for Stochastic optimization:
    
    def __init__( self,Xtrain):
        self.weights =np.ones((Xtrain.shape[1],1))
        #self.features = []  
    
#take all the features in the dataset    
        #for i in range (1,11):
        #    self.features.append(i)
        
    def learn(self, Xtrain, ytrain,lrate,alpha):
        thrh=0.00000001
        """ Learns using the traindata """
        """ The parameter alpha is user defined. When alpha=0, then the solution is OLS regrssion """
        temp=np.subtract(np.dot(Xtrain,self.weights),ytrain)
        temp1=np.multiply(temp,Xtrain.T)
        self.weights = np.subtract(self.weights,np.multiply(lrate,temp1))
        
        predict=np.dot(Xtrain,self.weights)        
        error=np.square(np.linalg.norm(np.subtract(predict,ytrain)))
        #print("The objective function is",error)
        
        if (error <=thrh):
           return False
        else:
           return True
        
    def bounding_weight(self):
        for i in range(0,self.weights.shape[0]):
            if self.weights[i]<=0.0001:
               self.weights[i]=0.0001
            elif self.weights[i] >=100:
               self.weights[i]=100
        
    def predict(self, Xtest):
        #Xless = Xtest[:,self.features]   
        #print ("Xtest dim is",Xtest.shape)   
        #print(self.weights)
        ytest = np.dot(Xtest, self.weights)   
        
        return ytest
        
class PRegression(Regressor):
    """
    Code for Poisson Regression 
    """
    def __init__( self,Xtrain):
        
        self.features=[1,2,3,4,5,6,7,8,9,10]
        #self.weights = np.ones((Xtrain.shape[1],1))
        self.weights = np.ones((11,1))
        #self.Cost=[]
        self.cost=[]
        Xless=Xtrain[:,self.features] 
        Xless = np.hstack((Xless, np.ones((Xless.shape[0],1))))
        #Xtrain1=Xtrain
        #print("The dimension of Xtrain is",Xtrain[0:1,0:Xtrain.shape[1]].shape)
        for i in range(0,Xtrain.shape[0]):
            
            a=math.exp(np.dot(self.weights.T,Xless[i:i+1,0:Xless.shape[1]].T))
            self.cost.append(a)
                
        self.Cost=np.diag(self.cost)
        print("The dimension of self.weights is",self.weights.shape)
        
    def feature_select(self,Xtrain,ytrain):
        
        Xtrain1=np.copy(Xtrain)
        Xtrain1=np.delete(Xtrain1,-1,1)
        hcor=[3,43,46,79,81,98,105,110,115,118,121,128,139,147,149,161,
              185,192,195,199,0,5,14,22,25,29,30,227,228]
        alfeat=[]
        print("The shape is",Xtrain1.shape)
        for i in range(0,Xtrain1.shape[1]):
            alfeat.append(i)
            #print(alfeat[i])
        for i in range(0,10):
            while(True):
              a=randint(0,229)
              if a in hcor:
                 continue
              else:
                  break
            self.features.append(a)
            
        print self.features
        
        self.weights = np.ones((10,1))
        #self.Cost=[]
        self.cost=[]
        Xless=Xtrain[:,self.features] 
        #Xtrain1=Xtrain
        #print("The dimension of Xtrain is",Xtrain[0:1,0:Xtrain.shape[1]].shape)
        for i in range(0,Xtrain.shape[0]):
            
            a=math.exp(np.dot(self.weights.T,Xless[i:i+1,0:Xless.shape[1]].T))
            #print ("The value of exponential is",a)
            self.cost.append(a)
        #print("The dimension of cost is",len(self.cost))    
        self.Cost=np.diag(self.cost)
        #print("The dimension of Cost is",self.Cost.shape)
        #print self.Cost
        #print("The dimension of self.weights is",self.weights.shape)
    
#take all the features in the dataset    
        #for i in range (1,11):
        #    self.features.append(i)
        
    def learn(self, Xtrain, ytrain,alpha):
        """ Learns using the traindata """
        """ The parameter alpha is user defined. When alpha=0, then the solution is OLS regrssion """
        thrh=0.00001
        Xless = Xtrain[:,self.features] 
        Xless = np.hstack((Xless, np.ones((Xless.shape[0],1))))
        temp1=np.linalg.inv(np.dot(np.dot(Xless.T,self.Cost),Xless))
        temp2=ytrain-self.cost
        temp3=np.dot(Xless.T,temp2)
        temp4 = np.reshape(temp3, (-1, 1))
        self.weights= self.weights + np.dot(temp1,temp4)   
        predict=np.dot(Xless,self.weights) 
        error=np.square(np.linalg.norm(np.subtract(predict,ytrain)))
        print("The objective function is",error)
        
        if (error <=thrh):
           return False
           
        self.cost=[]        
        for i in range(0,Xtrain.shape[0]):
            a=math.exp(np.dot(self.weights.T,Xless[i:i+1,0:Xless.shape[1]].T))
            self.cost.append(a)
        self.Cost=np.diag(self.cost)
        
        return True

        
            
    def predict(self, Xtest):
        Xless = Xtest[:,self.features] 
        Xless = np.hstack((Xless, np.ones((Xless.shape[0],1))))
        #print ("Xtest dim is",Xtest.shape)     
        #ytest = np.dot(Xtest, self.weights)  
        ytest = np.dot(Xless, self.weights)
        
        return ytest

 
    
    
    

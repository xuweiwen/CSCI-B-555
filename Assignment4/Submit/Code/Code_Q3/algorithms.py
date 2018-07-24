from __future__ import division  # floating point division
from scipy.cluster.vq import vq,kmeans,whiten
#from scipy.spatial.distance import pdist, squareform
import numpy as np
import utilities as utils

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
    def __init__( self, params=None ):
        self.weights = None
        self.features = [1,2,3,4,5,6,7,8,9,10]   
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:,self.features]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]        
        ytest = np.dot(Xless, self.weights)       
        return ytest
        
class RidgeRegression(Regressor):
    """
    Ridge Regression 
    """
    def __init__( self, params=None):
        self.weights = None
        self.features = []  
        self.regwt=0
        if params is not None and 'regwt' in params:
           self. regwt=params['regwt']
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        """ The parameter alpha is user defined. When alpha=0, then the solution is OLS regrssion """
        
        #print ("Xtrain dim is",Xtrain.shape)
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)+ np.multiply(self.regwt,np.identity(Xtrain.shape[1]))), Xtrain.T),ytrain)
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

class KernelRegression(Regressor):
    """
    Kernel Regression 
    """
    def __init__( self, params=None):
        self.weights = None
        self.features = []  
        """These are the default parameters"""
        self.regwt=0
        self.sigma=0.1
        self.function="linear"
        self.const=0
        
        if params is not None and 'regwt' in params:
           self. regwt=params['regwt']   
        self.function=params['method']  
        self.sigma=params['sigma']
        self.const=params['const']
        self.centroid=None
        self.newfeat=None
        self.testfeat=None
           
    def clustering(self,Xtrain):
        print("I am in the clustering function")
        print("The shape of Xtrain is",Xtrain.shape)
        col=Xtrain.shape[1]-1
        data=Xtrain[:,0:col]
        print("The shape of data is",data.shape)
        noc=400
        whitened = whiten(data)
        cluster=kmeans(whitened,noc)
        cluster=np.asarray(cluster[0])
        print("The shape of Cluster is",cluster.shape)
        self.centroid=cluster
        
    def build_new_feature(self,Xtrain,parm):
        print("The shape of centroid is",self.centroid.shape[0])
        print("The shape of Xtrain is",Xtrain.shape)
        col=Xtrain.shape[1]-1
        Xtrain=Xtrain[:,0:col]
        newfeat=[]
        for i in range(0,Xtrain.shape[0]):
            dotprod=[]
            for j in range (0,self.centroid.shape[0]):
                if self.function=="linear":
                   dotprod.append(np.dot(Xtrain[i],self.centroid[j]))
                elif self.function=="gaussian":
                     distance=utils.GaussianKernel(Xtrain[i],self.centroid[j],self.sigma)
                     dotprod.append(distance)
                elif self.function=="laplace":
                     #print("The value of sigma is",self.sigma)
                     distance=utils.LaplaceKernel(Xtrain[i],self.centroid[j],self.sigma)
                     dotprod.append(distance)
                elif self.function=="sigmoid":
                     alpha=1/Xtrain.shape[0]
                     distance=utils.SigmoidKernel(Xtrain[i],self.centroid[j],alpha,self.const)
                     dotprod.append(distance)
            newfeat.append(dotprod)
                #print("The value of newfeat is",self.newfeat[i][j])
        #print("The shape of self.newfeat is",self.newfeat.shape)    
        newfeat=np.asarray(newfeat)
        newfeat = np.hstack((newfeat, np.ones((newfeat.shape[0],1))))    
        if parm=="train":
           self.newfeat=newfeat
           print("The shape of self.newfeat is",self.newfeat.shape)
        else:
           self.testfeat=newfeat 
           print("The shape of self.testfeat is",self.testfeat.shape)
            
        
    def learn(self, Xtrain,ytrain):
        """ Learns using the traindata """
        """ The parameter self.regwt is user defined. When self.regwt=0, then the solution is OLS regrssion else it is regression with regularisation """
        
        #print ("Xtrain dim is",Xtrain.shape)
        if self.regwt==0:
            self.weights = np.dot(np.dot(np.linalg.inv(np.dot(self.newfeat.T,self.newfeat)), self.newfeat.T),ytrain)
        else:
            self.weights = np.dot(np.dot(np.linalg.inv(np.dot(self.newfeat.T,self.newfeat)+ np.multiply(self.regwt,np.identity(self.newfeat.shape[1]))), self.newfeat.T),ytrain)

        
    def predict(self, Xtest):
        ytest = np.dot(self.testfeat, self.weights)
        return ytest

    
    

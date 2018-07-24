import numpy as np
import math

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1) 

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))
    
def GaussianKernel(v1, v2, sigma):
    """returns the gaussian Kernel function"""
    return math.exp(-np.linalg.norm(np.subtract(v1,v2))**2/(2*sigma**2))

def LaplaceKernel(v1, v2, sigma):
    """returns the laplace Kernel function"""
    return math.exp(-np.linalg.norm(np.subtract(v1,v2))/(sigma))
    
def SigmoidKernel(v1, v2,alpha,const):
    """returns the tanh sigmoid Kernel function"""
    return math.tanh(alpha*np.dot(v1,v2)+const)
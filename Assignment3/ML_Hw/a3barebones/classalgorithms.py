from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
#from numpy import linalg as LA
#from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import LogisticRegression

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        
    def learn(self, Xtrain, ytrain,obj):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, params=None ):
        self.weights = None
    
        if params is not None and 'regwgt' in params:
            self.regwgt = params['regwgt']
        else:
            self.regwgt = 0.01
        
    def learn(self, Xtrain, ytrain,obj):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.regwgt*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes; need to complete the inherited learn and predict functions """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        """By default columns of ones will be used in the program"""
        self.usecolumnones = True
        if params is not None:
            self.usecolumnones = params['usecolumnones']
            
        """This table contains the prior probabilities of the two Class labels"""    
        self.prior_prob=[]
        
        """This dictionary contains the mu and sigma parameters learnt for every feature for every class label"""
        self.prob_table={}
        self.nof=0
        #self.model=GaussianNB()
        
    
    def learn(self, Xtrain, ytrain,obj):
        """ Learns using the traindata """
        """This part learns the prior of each class labels"""
        if self.usecolumnones==True:
            
           self.nof=Xtrain.shape[1]
        else:
           self.nof=Xtrain.shape[1]-1 
           
        postrain=Xtrain[ytrain==0]
        negtrain=Xtrain[ytrain==1]
        posprior=postrain.shape[0]/Xtrain.shape[0]
        negprior=negtrain.shape[0]/Xtrain.shape[0]
        self.prior_prob.extend((posprior,negprior))

        for i in range(0,self.nof):
            feature="Feature"+str(i)
            a={}
            for targDom in range(0,2):
                parameters = {}
                parameters["mu"] =  utils.mean(Xtrain[ytrain==targDom,i])
                parameters["sig"] = utils.stdev(Xtrain[ytrain==targDom,i])
                a[targDom]=parameters 
            self.prob_table[feature]=a
            
        """Python implementation of Naive Bayes"""
        
        #self.model.fit(Xtrain, ytrain)

            
    def predict(self, Xtest):
        
        ytest=[]
        
        for i in range(0,Xtest.shape[0]):
            feat_prob_all=[]  
            for k in range(0,2):
                feat_prob=[]
                for j in range(0,self.nof):
                    featname="Feature"+str(j)
                    mean=self.prob_table[featname][k]["mu"]
                    stdev=self.prob_table[featname][k]["sig"]
                    fprob=utils.calculateprob(Xtest[i,j], mean, stdev)
                    feat_prob.append(fprob)
                feat_prob_all.append(feat_prob)
            prob_pos=np.prod(np.array(feat_prob_all[0])) * self.prior_prob[0]
            prob_neg=np.prod(np.array(feat_prob_all[1])) * self.prior_prob[1]
            if prob_pos > prob_neg:
               ytest.append(0)
            else:
               ytest.append(1)
               
        ytest=np.asarray(ytest)       
        return ytest  
    
    """Inbuiilt Gaussian Naive Bayes Learning Model in Python"""    
    def predict1(self, Xtest):
        
        ytest= self.model.predict(Xtest)
        return ytest
        
            
    
class LogitReg(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """

    def __init__( self,params=None):
        self.weights = None
        self.prob=None
        self.Diag=None
        self.regwt=params['regwt']
        self.rtype=params['type']
        #self.model = LogisticRegression(C=self.regwt, penalty='l1', tol=0.01,fit_intercept=False)
                
    def learn(self, Xtrain, ytrain,obj):
        """ Learns using the traindata """
        """Initializes the weights from Logistic Regression weights"""
        self.weights=obj.weights
        thrh=0.00001
        yt = np.copy(ytrain)
        yt = np.reshape(yt,(-1,1))
        self.weights = np.reshape(self.weights,(-1,1))
        
        print ("The Regularisation parameter is",self.regwt)        
        j=0
        while(1):
            
          self.prob=None
          self.Diag=None
          prob=[]
          
          
          for i in range(0,Xtrain.shape[0]):
              Xrowvec = np.reshape(Xtrain[i],(-1,1))
              init_prod=np.dot(self.weights.T,Xrowvec)
              prob.append(init_prod)
              
          self.prob=utils.sigmoid(prob)
          self.Diag=np.diag(self.prob)
          self.prob=np.reshape(self.prob,(-1,1)) 
          Imat=np.subtract(np.identity(Xtrain.shape[0]),self.Diag)
          ymat=yt-self.prob
          prev_weights=self.weights
          
          """Vanilla Logistic Regression weight update rule"""
          
          if self.rtype== "None":
              
             print("Implementing Vanilla Logistic Regression") 
             self.weights= np.add(self.weights,np.dot(np.dot(np.linalg.inv(np.dot(np.dot(Xtrain.T,self.Diag),np.dot(Imat,Xtrain))),Xtrain.T),ymat))
          
          elif self.rtype=="L2":
              
               """Logistic Regression Regularisation weight update rule"""
               print("Implementing L2-regularised Logistic Regression") 
               regmat=self.regwt*np.identity(Xtrain.shape[1])
               invrsmat=np.linalg.inv(np.add(np.dot(np.dot(Xtrain.T,self.Diag),np.dot(Imat,Xtrain)),regmat))
               self.weights=np.add(self.weights,np.dot(invrsmat,np.subtract(np.dot(Xtrain.T,ymat),self.regwt*self.weights)))
          
          elif self.rtype=="L1":
              
               """Implementing Laso regularisation to Logistic Regression"""
               j=j+1
               print("Implementing L1 regularisation",j)
               invrsmat=np.linalg.inv(np.dot(np.dot(Xtrain.T,self.Diag),np.dot(Imat,Xtrain)))
               sign_vec=np.sign(self.weights)
               self.weights=np.add(self.weights,np.dot(invrsmat,np.subtract(np.dot(Xtrain.T,ymat),self.regwt*sign_vec))) 

          elif self.rtype=="Other":
               j=j+1
               """alpha is the probability for L1 regularisor and beta is the probability for L2 regularisor"""
               print("Implementing the regularisor of my choice")
               l1parm=self.regwt
               l2parm=1-self.regwt
               regmat=l2parm*np.identity(Xtrain.shape[1])
               invrsmat=np.linalg.inv(np.add(np.dot(np.dot(Xtrain.T,self.Diag),np.dot(Imat,Xtrain)),regmat))
               sign_vec=np.sign(self.weights)
               self.weights=np.add(self.weights,np.dot(invrsmat,np.subtract(np.subtract(np.dot(Xtrain.T,ymat),l2parm*self.weights),l1parm*sign_vec)))

                    
          """Calculate the distance between the old and the updated weight vector"""
          
          distance= np.linalg.norm(np.subtract(prev_weights,self.weights))
          print("The distance is",distance)
          
          if distance <= thrh:
             break
    
          if j==100:
             break
        #self.model.fit(Xtrain,ytrain) 

    def predict(self, Xtest):
        #print(self.weights)
        temp=self.weights[self.weights ==0]
        print("The shape of temp is",temp.shape)
        yvec = np.dot(Xtest, self.weights)
        ytest=utils.sigmoid(yvec)
        ytest[ytest >= 0.5] = 1     
        ytest[ytest < 0.5] = 0    
        return ytest

class NeuralNet(Classifier):
    """ Two-layer neural network; need to complete the inherited learn and predict functions """
    
    def __init__(self, params=None):
        # Number of input, hidden, and output nodes
        # Hard-coding sigmoid transfer for this implementation for simplicity
        self.ni = params['ni']
        self.nh = params['nh']
        self.no = params['no']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid

        # Set step-size
        self.stepsize = 0.01

        # Number of repetitions over the dataset
        self.reps = 5
        
        # Create random {0,1} weights to define features
        self.wi = np.random.randint(2, size=(self.nh, self.ni))
        self.wo = np.random.randint(2, size=(self.no, self.nh))

    def learn(self, Xtrain, ytrain,obj):
        """ Incrementally update neural network using stochastic gradient descent """       
        
        for reps in range(self.reps):
            for samp in range(Xtrain.shape[0]):
                xt= np.reshape(Xtrain[samp],(-1,1))
                hidden=self.transfer(np.dot(self.wi,xt))
                hidden = np.reshape(hidden,(-1,1))
                predicted=self.transfer(np.dot(self.wo,hidden))
                Del=(-ytrain[samp]*(1-predicted))+(1-ytrain[samp])*predicted
                grad_wo=np.dot(Del,hidden.T)
                grad_int=(self.wo*hidden.T*(1-hidden.T)).T
                xt=np.reshape(Xtrain[samp].T,(-1,1))
                grad_wi=Del*np.dot(grad_int,xt.T)
                self.update(Xtrain[samp,:],ytrain[samp],grad_wo,grad_wi)
            
    # Need to implement predict function, since currently inherits the default
            
    def predict(self, Xtest):
        
        
        ytest=[]
        for i in range(0,Xtest.shape[0]):
            (ah,ao)= self.evaluate(Xtest[i]) 
            if ao>0.5:
               ytest.append(1)
            else:
               ytest.append(0)
        return ytest       

    def evaluate(self, inputs):
        """ Including this function to show how predictions are made """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = np.ones(self.nh)
        ah = self.transfer(np.dot(self.wi,inputs))   

        # output activations
        ao = np.ones(self.no)
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)

    def update(self, inp, out,grado,gradi):
        """ This function needs to be implemented """
        self.wi= self.wi - self.stepsize*gradi
        self.wo=  self.wo - self.stepsize*grado
        return
        
class MyClassifier(Classifier):
    """ My own Classifier; need to complete the inherited learn and predict functions """

    def __init__( self,params=None):
        self.weights = None
        self.stepsize=0.00001
        #self.reps=5
    
    """Stochastic Gradient descent learning for My Classifier"""            
    def learn1(self, Xtrain, ytrain,obj):
        """ Learns using the traindata """
        """Initializes the weights from Logistic Regression weights"""
        self.weights = np.ones((Xtrain.shape[1],1))
        for reps in range(self.reps):
            for i in range(0,Xtrain.shape[0]):
                product=np.dot(self.weights.T,Xtrain[i])
                denom=math.sqrt(1+math.pow(product[0],2))            
                if denom<=1e-4:
                   denom=1e-4
                posprob=0.5*(1+product[0]/denom)
                negprob=0.5*(1-product[0]/denom)
                grad_int=posprob*negprob*(2*ytrain[i]-1)/denom
                self.weights=self.weights-self.stepsize*Xtrain[i]*grad_int            
            
    """Batch Gradient descent learning for My Classifier"""  
    def learn(self, Xtrain, ytrain,obj):
        """ Learns using the traindata """
        """Initializes the weights from Logistic Regression weights"""
        
        self.weights = np.ones((Xtrain.shape[1],1))
        prob=[]
        thrh=0.01
        for i in range(0,Xtrain.shape[0]):
            product=np.dot(self.weights.T,Xtrain[i])
            denom=math.sqrt(1+math.pow(product[0],2))
            if denom<=1e-4:
               denom=1e-4
            posprob=0.5*(1+product[0]/denom)
            negprob=0.5*(1-product[0]/denom)
            grad_int=posprob*negprob*(2*ytrain[i]-1)/denom
            prob.append(grad_int)

        prev_weights=self.weights
        prob=np.asarray(prob)
        prob=np.reshape(prob,(-1,1))

        while(1):
          
          grad=np.dot(Xtrain.T,prob)
          self.weights=np.add(self.weights,np.multiply(self.stepsize,grad))
          distance= np.linalg.norm(np.subtract(prev_weights,self.weights))
          print("The distance is",distance)
          if distance <= thrh:
             break;
            

    def predict(self, Xtest):
        ytest=[]
        for i in range(0,Xtest.shape[0]):
            product=np.dot(self.weights.T,Xtest[i])
            denom=math.sqrt(1+math.pow(product[0],2))
            if denom<=1e-4:
                   denom=1e-4
            posprob=0.5*(1+product[0]/denom)
            negprob=0.5*(1-product[0]/denom)
            if posprob>negprob:
               ytest.append(1)
            else:
               ytest.append(0)
        return ytest


        
            
    

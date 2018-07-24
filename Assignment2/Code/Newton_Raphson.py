#from decimal import *
from numpy import matrix
import numpy as np
from numpy.linalg import inv
thrh=0.000000000001
x=np.matrix([[-1.2,1]])
count=0 
while True:
  print x.item(0)
  print x.item(1)
   #print x
  j1=(400*(x.item(0)**3))-(400*x.item(0)*x.item(1))+(2*x.item(0))-2
  j2=200*(x.item(1)-x.item(0)**2)
  print j1,j2
  jac=np.matrix([[j1,j2]])
  #print jac[0][0]
  #print jac[0][1]
  h11=(1200*x.item(0)**2)-(400*x.item(1))+2
  h12=-400*x.item(0)
  h21=-400*x.item(0)
  h22=200
  hes=np.matrix([[h11,h12],[h21,h22]])
  hesinv = inv(hes)
  print hesinv
  print jac
  print("--------------------------------------")
  jhes=np.dot(jac,hesinv)
  print "jhes"
  print jhes
  xtemp=x-jhes
  print ("xtemp is",xtemp)
  print ("x is",x)
  if (0<= (abs(x-xtemp)).all() <=thrh):
      break
  count=count+1
  x=xtemp
print("The value of count is"+str(count))
print("The value of the matrix is")
print x

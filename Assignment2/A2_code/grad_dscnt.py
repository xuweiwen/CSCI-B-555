###Gradient descent implementation for Machine Learning Assignment#2
from decimal import *
import sys
sys.stdout=open("opt1.txt","w")
getcontext().prec = 25
x1_init=Decimal(-1.2)
x2_init=Decimal(1)
lrate=Decimal(0.001)
x1=x1_init
x2=x2_init
count=0
thrh=Decimal(0.000000000001)
grad_x1=Decimal(0)
grad_x2=Decimal(0)
while (True):
  cexp=Decimal(400*(x1**3))  
  grad_x1=cexp-(400*x1*x2)+(2*x1)-2
  print("The gradient along x1 is"+str(grad_x1))
  grad_x2=200*(x2-x1**2)
  print("The gradient along x2 is"+str(grad_x2))
  x1_new=x1-(lrate)*(grad_x1)
  x2_new=x2-(lrate)*(grad_x2)
  count=count+1
  print("The count is"+str(count))
  print("The new value of x1 is"+str(x1_new))
  print("The new value of x2 is"+str(x2_new))
  if ((0 <= abs(x1_new-x1) <= thrh) and (0 <= abs(x2_new- x2) <= thrh)):
      break
  x1=x1_new
  x2=x2_new
print("The diff is x_1 is"+str(abs(x1_new-x1)))
print("The diff is x_2 is"+str(abs(x2_new-x2)))
  
 
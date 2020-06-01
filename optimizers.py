import numpy as np
p=None
W=True
x=range
r=max
n=print
l=np.array
v=np.sqrt
t=np.zeros_like
class Optimizers():
 def __init__(q,o):
  q.all_weights=o
  q.g=t(o)
  q.g2=t(o)
  q.beta1=0.9
  q.beta2=0.999
  q.beta1t=1
  q.beta2t=1
 def sgd(q,error_f,gradient_f,fargs=[],n_epochs=100,learning_rate=0.001,error_convert_f=p,verbose=W):
  c=[]
  e=n_epochs//10
  for A in x(n_epochs):
   k=error_f(*fargs)
   K=gradient_f(*fargs)
   q.all_weights-=learning_rate*K
   if error_convert_f:
    k=error_convert_f(k)
   c.append(k)
   if verbose and((A+1)%r(1,e)==0):
    n(f'sgd: Epoch {epoch+1:d} Error={error:.5f}')
  return c
 def adam(q,error_f,gradient_f,fargs=[],n_epochs=100,learning_rate=0.001,error_convert_f=p,verbose=W):
  b=learning_rate 
  S=1e-8
  c=[]
  e=n_epochs//10
  for A in x(n_epochs):
   k=error_f(*fargs)
   K=gradient_f(*fargs)
   q.g[:]=q.beta1*q.g+(1-q.beta1)*K
   q.g2[:]=q.beta2*q.g2+(1-q.beta2)*K*K
   q.beta1t*=q.beta1
   q.beta2t*=q.beta2
   y=b*v(1-q.beta2t)/(1-q.beta1t)
   q.all_weights-=y*q.g/(v(q.g2)+S)
   if error_convert_f:
    k=error_convert_f(k)
   c.append(k)
   if verbose and((A+1)%r(1,e)==0):
    n(f'Adam: Epoch {A+1:d} Error={k:.5f}')
  return c
if __name__=='__main__':
 def parabola(d):
  return((w-d)**2)[0]
 def parabola_gradient(d):
  return 2*(w-d)
 w=l([0.0])
 D=Optimizers(w)
 d=5
 D.sgd(parabola,parabola_gradient,[d],n_epochs=100,learning_rate=0.1)
 n(f'sgd: Minimum of parabola is at {wmin}. Value found is {w}')
 w=l([0.0])
 D=Optimizers(w)
 D.adam(parabola,parabola_gradient,[d],n_epochs=100,learning_rate=0.1)
 n(f'adam: Minimum of parabola is at {wmin}. Value found is {w}')
# Created by pyminifier (https://github.com/liftoff/pyminifier)

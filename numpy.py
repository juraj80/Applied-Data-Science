
import numpy as np 
#CREATING ARRAYS
mylist=[1,2,3]
x=np.array(mylist)
print(x)
print()
 
y=np.array([4,5,6])
print(y)
print()
 
m=np.array([[7,8,9],[10,11,12]])
print (m.shape) 
print()
 
n=np.arange(0,30,2)# start at 0 count up by 2, stop before 30
n=n.reshape(3,5)# reshape array to be 3x5
print (n) 
print()
 
o=np.linspace(0,4,9)# return 9 evenly spaced values from 0 to 4
print (o)
print()
 
o.resize(3,3)
print(o)
print()
 
q=np.ones((3,2))
print(q)
 
d=np.diag(y)
print (y)
print()
 
k=np.array([1,2,3]*3)
print (k)
print()
 
l=np.repeat([1,2,3],3)
l=l.reshape(3,3)
print (l)
print ()
 
#COMBINING ARRAYS
p=np.ones([2,3],int)
print(p)
print()
 
p=np.vstack([p,2*p])
print(p)
print()
 
p=np.hstack([p,2*p])
print(p)
print()
 
#OPERATIONS
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**2)
print()
 
print(x.dot(y)) # dot product  1*4 + 2*5 + 3*6
print()
 
z=np.array([y,y**2])
print(z)
print(len(z))# number of rows of array
print(z.shape)
print()
print(z.T)
print()
print(z.dtype)
z=z.astype('f')
print(z.dtype)
print()
 
#MATH FUNCTIONS
a=np.array([-4,-2,-1,3,5])
print (a.sum())
print (a.min())
print (a.max())
print (a.mean())
print (a.std())
print (a.argmax())
print (a.argmin())
print()
 
#INDEXING/SLICING
s=np.arange(13)**2
print(s)
print()
print(s[0],s[4],s[-1])
print(s[1:5])
print(s[-4:])
print(s[-5::-2])
print()
 
r=np.arange(36)
r.resize((6,6))
print(r)
print(r[2,2]) 
print(r[3,3:6])
print(r[-1,::2])
print(r[r>30])
r[r>30]=30
print(r)
print()
 
#COPYING DATA
r2=r[:3,:3]
print(r2)
r2[:]=0 # Set this slice's values to zero ([:] selects the entire array)
print(r)
print()
 
r_copy=r.copy()
r_copy[:]=10
print(r_copy,'\n')
print(r)
print()
 
#ITERATING OVER ARRAYS
test=np.random.randint(0,10,(4,3))
print(test)
print()
 
for row in test:
  print(row)
print()
 
for i in range(len(test)):
  print (i,test[i])
print()
 
for i, row in enumerate(test):
  print('row',i,'is',row)
print()
 
test2=test**2 
print(test2)
print()
 
for i, j in zip(test,test2):
  print(i,'+',j,'=',i+j)
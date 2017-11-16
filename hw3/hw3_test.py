import csv
import sys
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import  Adam,Nadam,Adamax
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

def normalization(x):
	lx=len(x)
	temp=x.reshape(lx,48*48,1)
	sx=temp.std(axis=1)
	mx=temp.mean(axis=1)
#	print (mx.shape)
	sx+=0.00000000000001
	for i in range(len(x)):
		x[i]-=mx[i][0]
		x[i]/=sx[i][0]
	return x

xtest=np.zeros((7178,2304))

g=open(sys.argv[1],'r',encoding='big5')
reader2=csv.reader(g)
i=0
for row2 in reader2:
	if reader2.line_num==1 :
		continue
	temp=row2[1].strip('\n').split(' ')
	k=int(row2[0])
	xtest[i]=np.array(temp)
	i+=1

xt=xtest.reshape(7178,48,48,1)

##default

xt=normalization(xt)

model=load_model('save1.hdf5?dl=1')

batch=150
result=model.predict(xt,batch_size=batch)

output=result
yt=np.zeros(len(output))
for i in range(len(output)):
#	print (i)
	temp=0
	state=0
	for j in range(7):
		if output[i][j]>temp:
			temp=output[i][j]
			state=j
		else:
			continue
	yt[i]=state

with open(sys.argv[2],'w') as out:
	out.write('id,label\n')
	for i in range(len(yt)):
		out.write(str(i)+','+(str(yt[i])[0])+'\n')

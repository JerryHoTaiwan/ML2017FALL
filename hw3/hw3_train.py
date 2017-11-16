import csv
import numpy as np
import sys
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
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

def default(x):
	warning=set()
	for i in range(len(x)):
		std=np.std(x[i],axis=1)
		if std[6]<3 or std[20]<3 or std[30]<3 or std[10]<3 or std[40]<3 or std[25]<3:
			warning.add(i)
	return warning

def reverse(x):
	reverse=np.copy(x)
	for i in range(len(x)):
		reverse[i]=np.fliplr(x[i])
	mix=np.concatenate((x,reverse),axis=0)
	return mix


train=48418
valid=9000

ytable=np.zeros((28709,1))
xtable=np.zeros((28709,2304))

f=open(sys.argv[1],'r',encoding='big5')
reader=csv.reader(f)

i=0
m=0
shuffle=1

for row in reader:
	if reader.line_num==1 :
		continue
	temp=row[1].strip('\n').split(' ')
	k=int(row[0])
	ytable[i][0]=k
	xtable[i]=np.array(temp)
	i+=1

table=np.concatenate((ytable,xtable),axis=1)
if (shuffle):
	np.random.shuffle(table)
X=table[:,1:].reshape(28709,48,48,1).astype(float)
Y=np_utils.to_categorical(table[:,0],7).astype(int)

np.save('original',X)

MIX=reverse(X)
YR=np.concatenate((Y,Y),axis=0)

x=MIX[9000:]
xv=MIX[:9000]
y=YR[9000:]
yv=YR[:9000]

##default

bl=set()
bl2=set()
z1=len(bl)
z2=len(bl2)
bl=default(x)
bl2=default(xv)

xo=np.zeros((train-z1,48,48,1))
yo=np.zeros((train-z1,7))
xvo=np.zeros((valid-z2,48,48,1))
yvo=np.zeros((valid-z2,7))

j=0
for i in range(len(x)):
	if i not in bl:
		xo[j]=x[i]
		yo[j]=y[i]
		j+=1
j=0
for i in range(len(xv)):
	if i not in bl2:
		xvo[j]=xv[i]
		yvo[j]=yv[i]
		j+=1

xo=normalization(xo)
xvo=normalization(xvo)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

x=np.copy(xo)
y=np.copy(yo)
xv=np.copy(xvo)
yv=np.copy(yvo)
batch=150
"""
x/=2
xv/=2
xt/=2
"""
datagen.fit(x)


model=Sequential()
model.add(Convolution2D(64,(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))

model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.4))

model.add(Convolution2D(256,(3,3),activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())
#model.add(Dropout(0.4))

model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=7))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer="Adamax",loss='categorical_crossentropy',metrics=['accuracy'])

ep=689

model.fit_generator(datagen.flow(x,y, batch_size=batch),steps_per_epoch=len(x)/batch, epochs=ep)
ev=model.evaluate(xv,yv,batch_size=batch)
print (' ',ev[1],'\n')

print ('save')
model.summary()

model.save('save.h5')


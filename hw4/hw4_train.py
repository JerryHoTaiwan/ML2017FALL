#-*- coding: utf-8 -*-
import numpy as np
import pickle
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,GRU
from keras.optimizers import  Adam,Nadam,Adamax
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from gensim.models import word2vec
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys

def word2index(model,text,num):
        vocab = dict([(k,v.index) for k,v in model.wv.vocab.items()]);
        matrix = np.zeros((num,32))
        for i in range(num):
                line = text[i]
                j = 0;
                for word in line.split(' '):
                        if j > 31: #padding(limit) sequences to lenght 32
                                break
                        if word in vocab:
                                matrix[i][j] = vocab[word]
                        j+=1
        return matrix

def read_unlabel(file):
        f = open(file,'r')
        x = list()
        for line in f.readlines():
                x.append(line[:-1])
        f.close()
        return np.array(x)

def read_label(file):
    f = open(file,'r')
    x = list()
    for line in f:
        x.append(line[10:])
    f.close()
    return np.array(x)

def read_ans(file):
    f = open(file,'r')
    y = list()
    for line in f:
        y.append(line[0])
    f.close()
    return np.array(y)

count = 40
ep = 5
embed_model = word2vec.Word2Vec.load("embed_model_128")

weights = np.array(embed_model.wv.syn0)

X = read_label(sys.argv[1])
XU = read_unlabel(sys.argv[2])
X = word2index(embed_model,X,200000)
XU = word2index(embed_model,XU,1000000)
Y = read_ans(sys.argv[1])
ss = 1
print (X.shape,XU.shape,Y.shape)

#print (weights.shape,len(weights))
valid_size = 10000
X_train = X[valid_size:]
Y_train = Y[valid_size:]
X_valid = X[:valid_size]
Y_valid = Y[:valid_size]
vocab_size = weights.shape[0]

model = Sequential()
model.add(Embedding(vocab_size,128,input_length=32,weights=[weights],trainable=True))
model.add(GRU(units=260,activation="selu",dropout=0.2,recurrent_dropout=0.2,kernel_initializer='lecun_normal',return_sequences=True))
model.add(GRU(units=260,activation="selu",dropout=0.2,recurrent_dropout=0.2,kernel_initializer='lecun_normal',return_sequences=True))
model.add(GRU(units=250,activation="selu",dropout=0.2,recurrent_dropout=0.2,kernel_initializer='lecun_normal',return_sequences=True))
model.add(GRU(units=250,activation="selu",dropout=0.2,recurrent_dropout=0.2,kernel_initializer='lecun_normal'))
model.add(Dense(units=128,activation='selu',kernel_initializer='lecun_normal'))
#model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid',kernel_initializer='lecun_normal'))
model.compile(loss='binary_crossentropy',optimizer='Adamax',metrics=['accuracy'])
model.summary()

while (1):

    if (count==0):
        break
    model.fit(X,Y,validation_split=0.2,epochs=ep,batch_size=128,shuffle=True)
    ev = model.evaluate(X_valid,Y_valid,batch_size=128)
    print (ev[1])
    count-=ep
    save = 1
    
    if (save):
        model.save('output.h5')

    ss = 1
    if (ss):
        np.random.shuffle(XU)
        semi = XU[:200000]
        predict = model.predict(semi).reshape(200000)
        small = np.where(predict<0.1)
        large = np.where(predict>0.9)
        smallx = semi[small]
        largex = semi[large]
        smally = np.round(predict[small])
        largey = np.round(predict[large])
        X = np.concatenate((X,smallx,largex),axis=0)
        Y = np.concatenate((Y,smally,largey),axis=0)
        print (Y.shape)

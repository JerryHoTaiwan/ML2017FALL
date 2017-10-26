import os, sys
import pandas as pd
import numpy as np
from random import shuffle
import argparse
import csv
import sys

def sigmoid(z):
	z2=np.ones(len(z))/(np.ones(len(z))+np.exp(-1*z))
	return np.clip(z2,0.00000000000001,0.99999999999999)

def valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.pinv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
#    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test


def train(X_train, Y_train):
    
    # Gaussian distribution parameters
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((106,))
    mu2 = np.zeros((106,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((106,106))
    sigma2 = np.zeros((106,106))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
    N1 = cnt1
    N2 = cnt2

    return (mu1, mu2, shared_sigma, N1, N2)

def infer(X_test, mu1,mu2,shared_sigma,N1,N2):

    # Predict
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)

    return y_

#read the file
f=open(sys.argv[3],'r',encoding='big5')
reader=csv.reader(f)
g=open(sys.argv[4],'r',encoding='big5')
reader2=csv.reader(g)

x=[]
y=[]
xt=[]
yt=[]

for row in reader:
	if reader.line_num==1: #or reader.line_num>10000:
		continue
	x.append(row)

for row in reader2:
	if reader2.line_num==1: #or reader2.line_num>10000:
		continue
	y.append(float(row[0]))

x=np.array(x,dtype=float)
y=np.array(y,dtype=float)
#writing
t=open(sys.argv[5],'r',encoding='big5')
reader3=csv.reader(t)
for row in reader3:
	if reader3.line_num==1:
		continue
	xt.append(row)

#testing
xt=np.array(xt,dtype=float)

####training#####
(x,xt)=normalize(x,xt)
(m1,m2,sig,n1,n2)=train(x,y)
valid(x,y,m1,m2,sig,n1,n2)
yt=infer(xt,m1,m2,sig,n1,n2)
print (len(yt),len(y))
# write
with open(sys.argv[6],'w') as output:
	output.write('id,label\n')
	for i in range(len(yt)):
		output.write(str(i+1)+','+(str(yt[i])[0])+'\n')

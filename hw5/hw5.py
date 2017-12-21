import pandas as pd
import numpy as np
import pickle
import csv
import sys
from keras.models import Sequential,load_model

def read_data(file):
        f = open(file,'r')
        test = np.array(pd.read_csv(file)).astype(str)
        data = np.array([line for line in csv.reader(f)])
        return data[1:,1:]

def denormalize(x):
	return x*1.11689766115+3.58171208604

test = read_data(sys.argv[1])

with open('movie_tok.pickle', 'rb') as handle:
    tok_m = pickle.load(handle)
with open('user_tok.pickle', 'rb') as handle:
    tok_u = pickle.load(handle)

users = test[:,0]
movies = test[:,1]

u_test = np.array(tok_u.texts_to_sequences(users))
m_test = np.array(tok_m.texts_to_sequences(movies))

model = load_model('output.h5')
model3 = load_model('output3.h5')
model5 = load_model('output5.h5')
model_20 = load_model('num_20.h5')

result1 = (denormalize(model.predict([u_test,m_test,np.zeros(len(u_test))],batch_size=128)))
result3 = (denormalize(model3.predict([u_test,m_test],batch_size=128)))
result5 = (denormalize(model5.predict([u_test,m_test],batch_size=128)))
result_20 = (denormalize(model_20.predict([u_test,m_test],batch_size=128)))

result = (result1+result3+result_20+result5)/4

with open(sys.argv[2],'w') as out:
	out.write('TestDataID,Rating\n')
	for i in range(len(result)):
		out.write(str(i+1)+','+(str(float(result[i])))+'\n')
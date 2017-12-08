import numpy as np
from keras.models import Sequential,load_model
from gensim.models import word2vec
import sys

def read_test(file):
        x = list()
        count = 0
        f = open(file,'r')
        for line in f:
            if count == 0: pass
            elif count <= 10: x.append(line[2:][:-1])
            elif count <= 100: x.append(line[3:][:-1])
            elif count <= 1000: x.append(line[4:][:-1])
            elif count <= 10000: x.append(line[5:][:-1])
            elif count <= 100000: x.append(line[6:][:-1])
            else: x.append(line[7:][:-1])
            count += 1
        return np.array(x)

def word2index(model,text):
        num = 200000
        vocab = dict([(k,v.index) for k,v in model.wv.vocab.items()]);
        matrix = np.zeros((num,32))
        for i in range(num):
                line = text[i]
                j = 0;
                for word in line.split(' '):
                        if j > 31:
                                break
                        if word in vocab:
                                matrix[i][j] = vocab[word]
                        j += 1
        return matrix

embed_model = word2vec.Word2Vec.load("embed_model_128")

model5 = load_model('test5.h5')
model6 = load_model('test6_2.h5')

xt = read_test(sys.argv[1])
#print (xt[0])
Xt = word2index(embed_model,xt)
#print (xt[666],Xt[666])

result5 = model5.predict(Xt,batch_size=128)
result6 = model6.predict(Xt,batch_size=128)
y = np.round(np.array(0.5*result6+0.5*result5)).astype(int)

with open(sys.argv[2],'w') as out:
	out.write('id,label\n')
	for i in range(len(y)):
		out.write(str(i)+','+(str(y[i])[1])+'\n')
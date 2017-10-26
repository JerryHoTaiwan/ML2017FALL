import numpy as np
import csv
import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.externals import joblib

model=joblib.load('best.pkl')

f=open(sys.argv[3],'r',encoding='big5')
reader=csv.reader(f)

t=open(sys.argv[5],'r',encoding='big5')
reader3=csv.reader(t)


x=[]
xt=[]
yt=[]

for row in reader:
	if reader.line_num==1: #or reader.line_num>10000:
		continue
	x.append(row)

for row in reader3:
	if reader3.line_num==1: #or reader.line_num>10000:
		continue
	xt.append(row)

x=np.array(x,dtype=float)
xt=np.array(xt,dtype=float)
yt=np.zeros((len(xt)))
X = np.concatenate((x, xt))
m=X.mean(axis=0)
s=X.std(axis=0)
#normalization
mx=x.mean(axis=0)
sx=x.std(axis=0)
mxt=xt.mean(axis=0)
sxt=xt.std(axis=0)

#normalization
for i in range(len(x)):
	x[i][1]=0
	for j in range(len(x[0])):
		x[i][j]=(x[i][j]-m[j])/s[j]

for i in range(len(xt)):
	xt[i][1]=0
	for j in range(len(xt[0])):
		xt[i][j]=(xt[i][j]-m[j])/s[j]

yt=model.predict(xt)

# write
with open(sys.argv[6],'w') as output:
	output.write('id,label\n')
	for i in range(len(yt)):
		output.write(str(i+1)+','+(str(yt[i])[0])+'\n')
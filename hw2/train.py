import numpy as np
import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.externals import joblib

f=open('X_train','r',encoding='big5')
reader=csv.reader(f)
g=open('Y_train','r',encoding='big5')
reader2=csv.reader(g)
t=open('X_test','r',encoding='big5')
reader3=csv.reader(t)


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

for row in reader3:
	if reader3.line_num==1: #or reader.line_num>10000:
		continue
	xt.append(row)

x=np.array(x,dtype=float)
y=np.array(y,dtype=float)
xt=np.array(xt,dtype=float)
yt=np.zeros((len(xt)))
X = np.concatenate((x, xt))
m=X.mean(axis=0)
s=X.std(axis=0)
#normalization
mx=x.mean(axis=0)
sx=x.std(axis=0)
for i in range(len(x)):
	x[i][1]=0
	for j in range(len(x[0])):
		x[i][j]=(x[i][j]-m[j])/s[j]
#y = np_utils.to_categorical(y,2)

#normalization
mxt=xt.mean(axis=0)
sxt=xt.std(axis=0)
#default
for i in range(len(xt[0])):
	if sxt[i]==0:
		sxt[i]+=0.0000001
for i in range(len(xt)):
	xt[i][1]=0
	for j in range(len(xt[0])):
		xt[i][j]=(xt[i][j]-m[j])/s[j]

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME.R",
                         n_estimators=1200)

bdt.fit(x, y)

#save the model
joblib.dump(bdt,'best.pkl')


yt=bdt.predict(xt)

# write
with open('sk.csv','w') as output:
	output.write('id,label\n')
	for i in range(len(yt)):
		output.write(str(i+1)+','+(str(yt[i])[0])+'\n')

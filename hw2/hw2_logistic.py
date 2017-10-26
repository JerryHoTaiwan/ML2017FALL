import csv
import numpy as np
import sys

classA={' Yugoslavia', ' England', ' Iran', ' France', ' India', ' Taiwan', ' Canada', ' Cambodia', ' Japan', ' Italy'}
classB={' United-States', '?_native_country', ' Cuba', ' Greece', ' China', ' Ireland', ' Philippines', ' Germany', ' Hungary', ' Hong', ' Scotland'}
classC={' Laos', ' Ecuador', ' South', ' Haiti', ' El-Salvador', ' Portugal', ' Trinadad&Tobago', ' Puerto-Rico', ' Jamaica', ' Thailand', ' Poland'}
classD={' Holand-Netherlands', ' Guatemala', ' Dominican-Republic', ' Mexico', ' Honduras', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Vietnam', ' Nicaragua', ' Columbia'}
country=[]

we=open('weight.csv','r',encoding='big5')
reader4=csv.reader(we)
w=[]
for row in reader4:
	w.append(row[0])

f=open(sys.argv[3],'r',encoding='big5')
reader=csv.reader(f)
for row in reader:
	i=reader.line_num
	if reader.line_num==1: #or reader.line_num>10000:
		for i in range(len(row)):
			if i in range(64,106):
				country.append(row[i])
		continue

t=open(sys.argv[5],'r',encoding='big5')
reader3=csv.reader(t)
xt=np.zeros((16281,68))
for row in reader3:
	i=reader3.line_num
	if reader3.line_num==1: #or reader.line_num>10000:
		continue
	for j in range(64):
		xt[i-2][j]=row[j]
	for k in range(64,106):
		if row[k]=='1':
			if country[k-64] in classA:
				xt[i-2][64]=1
				continue
			if country[k-64] in classB:
				xt[i-2][65]=1
				continue
			if country[k-64] in classC:
				xt[i-2][66]=1
				continue
			if country[k-64] in classD:
				xt[i-2][67]=1
				continue
#normalization
mxt=xt.mean(axis=0)
sxt=xt.std(axis=0)

#default
for i in range(len(xt[0])):
	if sxt[i]==0:
		sxt[i]+=0.0000001

for i in range(len(xt)):
	for j in range(len(xt[0])):
		xt[i][j]=(xt[i][j]-mxt[j])/sxt[j]

w=np.array(w,dtype=float)
b=-2.09944611817
xt=np.array(xt,dtype=float)
yt=[]
yt=xt.dot(w)+b
for i in range(len(yt)):
	if yt[i]>0:
		yt[i]=1
	else:
		yt[i]=0
with open(sys.argv[6],'w') as output:
	output.write('id,label\n')
	for i in range(len(yt)):
		output.write(str(i+1)+','+(str(yt[i])[0])+'\n')
import csv
import numpy as np

def sigmoid(z):
	z2=np.ones(len(z))/(np.ones(len(z))+np.exp(-1*z))
	return z2

def crossentropy(f,y):
	f2=np.ones(len(f))-f
	i=0
	for i in range(len(f2)):
		if f2[i]==0:
			print(f2[i],f[i])
			i+=1
	f=np.log(f)
	f2=np.log(f2)
	y2=np.ones(len(y))-y
	cross=-1*(f.dot(y)+f2.dot(y2))
	return cross

#needed arrays and sets for the data
country=[]
x=np.zeros((32561,68))
y=[]
xt=np.zeros((16281,68))
yt=[]

classA={' Yugoslavia', ' England', ' Iran', ' France', ' India', ' Taiwan', ' Canada', ' Cambodia', ' Japan', ' Italy'}
classB={' United-States', '?_native_country', ' Cuba', ' Greece', ' China', ' Ireland', ' Philippines', ' Germany', ' Hungary', ' Hong', ' Scotland'}
classC={' Laos', ' Ecuador', ' South', ' Haiti', ' El-Salvador', ' Portugal', ' Trinadad&Tobago', ' Puerto-Rico', ' Jamaica', ' Thailand', ' Poland'}
classD={' Holand-Netherlands', ' Guatemala', ' Dominican-Republic', ' Mexico', ' Honduras', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Vietnam', ' Nicaragua', ' Columbia'}

#read the file
f=open('X_train','r',encoding='big5')
reader=csv.reader(f)
g=open('Y_train','r',encoding='big5')
reader2=csv.reader(g)

for row in reader:
	i=reader.line_num
	if reader.line_num==1: #or reader.line_num>10000:
		for i in range(len(row)):
			if i in range(64,106):
				country.append(row[i])
		continue
	for j in range(64):
		x[i-2][j]=row[j]
	for k in range(64,106):
		if row[k]=='1':
			if country[k-64] in classA:
				x[i-2][64]=1
				continue
			if country[k-64] in classB:
				x[i-2][65]=1
				continue
			if country[k-64] in classC:
				x[i-2][66]=1
				continue
			if country[k-64] in classD:
				x[i-2][67]=1
				continue

for row in reader2:
	if reader2.line_num==1: #or reader2.line_num>10000:
		continue
	y.append(float(row[0]))

x=np.array(x,dtype=float)
y=np.array(y,dtype=float)
#normalization
mx=x.mean(axis=0)
sx=x.std(axis=0)
for i in range(len(x)):
	for j in range(len(x[0])):
		x[i][j]=(x[i][j]-mx[j])/sx[j]

#needed arrays for training
w=np.zeros(len(x[0]))
gw=np.zeros(len(x[0]))
rw=np.zeros(len(x[0]))
f=np.zeros(len(x[0]))
b=0
gb=0
rb=0

####training#####

n=10000
rate=0.1
reg=0.01
print ('start training')

for i in range(n):

	f=sigmoid(x.dot(w)+b)
	#default
	for j in range(len(f)):
		if f[j]>0.9999999999:
			f[j]=0.9999999999
	gb = np.sum(f-y)
	gw = x.T.dot(f-y)+reg*w
	rw += gw*gw
	rb += gb*gb
	w -= (gw/np.sqrt(rw))*rate
	b -= (gb/np.sqrt(rb))*rate
	if i%1000==0:
		print(i,crossentropy(f,y))

######training end#######

t=open('X_test','r',encoding='big5')
reader3=csv.reader(t)

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

#testing
xt=np.array(xt,dtype=float)

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
yt=xt.dot(w)+b

for i in range(len(yt)):
	if yt[i]>0:
		yt[i]=1
	else:
		yt[i]=0

# write
with open('country.csv','w') as output:
	output.write('id,label\n')
	for i in range(len(yt)):
		output.write(str(i+1)+','+(str(yt[i])[0])+'\n')

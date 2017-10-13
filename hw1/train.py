import csv
import numpy as np

#read the file
f=open('train.csv','r',encoding='big5')
n=100000
rate=0.3
feature=10
y=[]
x=[]
for i in range(0,feature):
	x.append([])

k=0
r=0
reader=csv.reader(f)

for row in reader:
	#cancel the first line and try to select partial data
	if reader.line_num==1:
		continue
	#input original x
	count=k%feature
#	print (count,row[2])
	if row[2]=='PM10' or row[2]=='PM2.5' or row[2]=='CO' or row[2]=='O3' or row[2]=='RAINFALL' or row[2]=='SO2' or row[2]=='WIND_DIREC' or row[2]=='WIND_SPEED' or row[2]=='WD_HR' or row[2]=='WS_HR':
#		print (count,row[2])
		for i in range(3,27):
			if row[i]=='NR':
				row[i]=0
			x[count].append(row[i])
		k+=1

	if row[2]=='PM2.5':
		#put things into y
		if row[0][-1]=='1' and row[0][-2]=='/':
			for i in range(12,27):
				y.append(row[i])
		else:
			for i in range(3,27):
				y.append(row[i])

#consider the wind wow 666666666
wcos = []
wsin = []
whcos = []
whsin = []
for i in range(len(x[0])):
	x[6][i]=float(x[6][i])*6.28/360
	x[8][i]=float(x[8][i])*6.28/360
#	print (type(x[6][i]))
	wcos.append(float(x[7][i])*np.cos(x[6][i]))
	wsin.append(float(x[7][i])*np.sin(x[6][i]))	
	whcos.append(float(x[9][i])*np.cos(x[8][i]))
	whsin.append(float(x[9][i])*np.sin(x[8][i]))	
	x[6][i]=wcos[i]
	x[7][i]=wsin[i]
	x[8][i]=whcos[i]
	x[9][i]=whsin[i]

#update x and consider the square of x
xn=[]
xn2=[]
for i in range(0,len(y)):
	#months here will be smaller than exactly ones by 0
	xn.append([])
	xn2.append([])
#	print (len(xn))
	#xn count from xn[0]
	month=int((i)/471)
	hour=i%471
	for j in range(0,feature):
		for k in range(480*month+hour,480*month+hour+9):
#			print (i,j,k)
			xn[i].append(x[j][k])
			if j<6:
				xn2[i].append(float(x[j][k])**2)
#print (len(xn),len(xn[0]))
#turn the type into array and create the weighted matrix, the gradient matrix, and each learning rate
#regularization
xa1=np.array(xn,dtype=float)
xa2=np.array(xn2,dtype=float)
xa=np.concatenate((xa1,xa2),axis=1)
ya=np.array(y,dtype=float)
xo=np.concatenate((xa1,xa2),axis=1)
yo=np.array(y,dtype=float)

w=[]
gr=[]
r=[]
reg=0

for i in range(len(xa[0])):
	w.append(0)
	gr.append(0)
	r.append(0)
wa=np.array(w,dtype=float)
ga=np.array(w,dtype=float)
ra=np.array(w,dtype=float)
#normalization
mx=xa.mean(axis=0)
sx=xa.std(axis=0)
my=ya.mean(axis=0)
sy=ya.std(axis=0)
for i in range(len(xa)):
	ya[i]=(yo[i]-my)/sy
	for j in range(len(xa[0])):
		xa[i][j]=(xa[i][j]-mx[j])/sx[j]

#cheating
(answer,limit,c,dZ)=np.linalg.lstsq(xo,yo)
print (limit)

#####training#####
print ('start training')
loss=0

# a more effictive way? try to apply 
for i in range(0,n):
#	loss = (ya-xa.dot(wa)).dot((ya-xa.dot(wa)))
#	for j in range(len(ya)):
#		ga=(xa[j].dot(wa)-ya[j])*xa[j]+reg*wa
#		ra+=ga*ga
	if (i%10000)==0:
		print (i)
	ga = xa.T.dot(xa.dot(wa)-ya)+reg*wa
	ra += ga*ga
	wa -= (ga/np.sqrt(ra))*rate
	loss = (ya-xa.dot(wa)).dot((ya-xa.dot(wa)))
#	print (loss)

#print (wa)
#put the normalized value back and calculate
loss=0

b=my-sum(mx*wa*sy/sx)
wa=sy*wa/sx
l=(yo-xo.dot(wa))
for i in range(l.size):
	l[i]-=b
loss=l.dot(l)
print (loss,b)
#print (loss,np.sqrt(loss/len(y)),b)

#testing data
xt=np.zeros((240,len(wa)))
j=0
k=0
g=open('test.csv','r',encoding='big5')
reader2=csv.reader(g)
#print (len(xt[0]))
for row in reader2:
	if k==9*feature:
		j+=1
		k=0
#	print (len(wa))
	if row[1]=='PM10' or row[1]=='PM2.5' or row[1]=='CO' or row[1]=='O3' or row[1]=='RAINFALL' or row[1]=='SO2' or row[1]=='WIND_DIREC' or row[1]=='WIND_SPEED'  or row[1]=='WD_HR' or row[1]=='WS_HR':
		for i in range(2,11):
			if row[i]=='NR':
				row[i]=0
#			print (k,row[1])
			xt[j][k]=row[i]
			if row[1]!='WIND_DIREC' and row[1]!='WIND_SPEED' and row[1]!='WD_HR' and row[1]!='WS_HR':
				xt[j][k+9*feature]=float(row[i])**2
			k+=1
#consider the wind, notice that cos and sin should'nt hove 2nd power term
wtcos = []
wtsin = []
whtcos = []
whtsin = []
for i in range(240):
	wtcos = []
	wtsin = []
	whtcos = []
	whtsin = []
	for j in range(9):
		xt[i][54+j]=float(xt[i][54+j])*6.28/360
		xt[i][72+j]=float(xt[i][72+j])*6.28/360
#	print (type(x[6][i]))
		wtcos.append(float(xt[i][63+j])*np.cos(xt[i][54+j]))
		wtsin.append(float(xt[i][63+j])*np.sin(xt[i][54+j]))
		whtcos.append(float(xt[i][81+j])*np.cos(xt[i][72+j]))
		whtsin.append(float(xt[i][81+j])*np.sin(xt[i][72+j]))	
		xt[i][54+j]=wtcos[j]
		xt[i][63+j]=wtsin[j]
#		print (i,j)
		xt[i][72+j]=whtcos[j]
		xt[i][81+j]=whtsin[j]

#const=np.ones((len(wa),1))
yt=(xt.dot(wa))
for i in range(len(yt)):
	yt[i]+=b

ans=np.array(answer)
yta=(xt.dot(ans))
print (wa,ans)
np.savetxt('w07.csv',wa,delimiter=',')

# write
with open('output07.csv','w') as output:
	output.write('id,value\n')
	for i in range(240):
		output.write('id_'+str(i)+','+str(yt[i])+'\n')

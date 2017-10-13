import csv
import sys
import numpy as np

f=open('w6.csv','r',encoding='big5')
reader=csv.reader(f)
w=np.zeros(126)
i=0
for row in reader:
	w[i]=float(row[0])
	i+=1

xt=np.zeros((240,len(w)))
b=-2.7070254168000005

g=open(sys.argv[1],'r',encoding='big5')
reader2=csv.reader(g)
feature=8
j=0
k=0
for row in reader2:
	if k==9*feature:
		j+=1
		k=0
#	print (len(wa))
	if row[1]=='PM10' or row[1]=='PM2.5' or row[1]=='CO' or row[1]=='O3' or row[1]=='RAINFALL' or row[1]=='SO2' or row[1]=='WIND_DIREC' or row[1]=='WIND_SPEED':
		for i in range(2,11):
			if row[i]=='NR':
				row[i]=0
#			print (k,row[1])
			xt[j][k]=row[i]
			if row[1]!='WIND_DIREC' and row[1]!='WIND_SPEED':
				xt[j][k+9*feature]=float(row[i])**2
			k+=1
#consider the wind, notice that cos and sin should'nt hove 2nd power term
wtcos = []
wtsin = []
for i in range(240):
	wtcos = []
	wtsin = []
	for j in range(9):
		xt[i][54+j]=float(xt[i][54+j])*6.28/360
#	print (type(x[6][i]))
		wtcos.append(float(xt[i][63+j])*np.cos(xt[i][54+j]))
		wtsin.append(float(xt[i][63+j])*np.sin(xt[i][54+j]))	
		xt[i][54+j]=wtcos[j]
		xt[i][63+j]=wtsin[j]
yt=(xt.dot(w))
for i in range(len(yt)):
	yt[i]+=b

# write
with open(sys.argv[2],'w') as output:
	output.write('id,value\n')
	for i in range(240):
		output.write('id_'+str(i)+','+str(yt[i])+'\n')

print ('success')
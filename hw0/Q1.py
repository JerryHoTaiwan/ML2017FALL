import sys

f=open(sys.argv[1],'r')
output=open('Q1.txt','w')
st=f.read()
st2=st.strip("\n").split(" ")
result=[]
for i in st2:
	if i not in result:
		result.append(i)
#print result

for j in result:
	output.write(j+" "+str(result.index(j))+" "+str(st2.count(j))+'\n')

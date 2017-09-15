from PIL import Image
import sys
im=Image.open(sys.argv[1])
w,h=im.size
output=Image.new("RGB",(w,h))
#print (im.size)

for x in range(0,w):
	for y in range(0,h):
		rgb=im.getpixel((x,y))
		pix=[]
		nR=int(rgb[0]/2)
		nG=int(rgb[1]/2)
		nB=int(rgb[2]/2)
		pix.append((nR,nG,nB))
		output.putpixel((x,y),pix[0])
output.save('Q2.png')

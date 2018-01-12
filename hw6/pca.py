from skimage import io,transform
import numpy as np
import os
import sys
size = 600
k = 4
img_path = sys.argv[1]+'/'+sys.argv[2]
print (os.path.join(sys.argv[1]+'/'+sys.argv[2]))

def toimg(img):
        img = img * 1.0
        img -= np.min(img)
        img /= np.max(img)
        img = (img * 255).astype(np.uint8)
        return img

def reconstruct(img):
	global eigenfaces,k,img_mean
	print (X_mean.dtype)
	eigen = np.copy(eigenfaces)
	f_img = img.flatten()-img_mean.flatten()
	weight = np.zeros(k)
	for i in range(k):
		weight[i] = np.dot(f_img,eigen[:,i])
	d_img = np.dot(weight,eigen.T).reshape(3*size*size,1)
#	print (weight[6])
	re_img = toimg(d_img+X_mean).reshape(size,size,3)
#	re_img = toimg(d_img).reshape(size,size,3)+toimg(X_mean).reshape(size,size,3)
	return re_img

img_mean = np.load('img_face_nr.npy')
X_mean = img_mean.reshape(size*size*3,1)
eigenfaces = np.load('eigenfaces_nr.npy')
img = io.imread(img_path)
new_img = img.flatten()
re_img = reconstruct(new_img)
#io.imsave('original.jpg',img)
io.imsave('reconstruction.jpg',re_img)
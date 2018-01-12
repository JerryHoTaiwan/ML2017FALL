import numpy as np
from skimage.io import ImageCollection, imsave, imshow, imread
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys



img_data = np.load(sys.argv[1])
test_file = sys.argv[2]
out_file = sys.argv[3]

valid_size = 10000
encoding_dim = 2
PATIENCE = 10

all_labels = np.load('pca_labels.npy')
#testing
test_data = pd.read_csv( test_file ,sep=",",engine="python",dtype='U').values
x_test1 = test_data[:,1].astype(int)
x_test2 = test_data[:,2].astype(int)
result = []
for i in range(x_test1.shape[0]):
        if all_labels[x_test1[i]] == all_labels[x_test2[i]]:
                result.append(1)
        else:
                result.append(0)

print("Generating resuls ... ")
output_file = open(out_file,'w')
writer = csv.writer(output_file)

writer.writerow(["ID","Ans"])

for i in range(len(result)):
        writer.writerow([i,result[i]])

output_file.close()

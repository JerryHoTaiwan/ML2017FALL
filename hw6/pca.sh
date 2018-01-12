#!/bin/bash
wget -O 'eigenfaces_nr.npy' 'https://www.dropbox.com/s/kuoajgmplu9xiy9/eigenfaces_nr.npy?dl=1'
python3 pca.py $1 $2 

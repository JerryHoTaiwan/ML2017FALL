#!/bin/bash
wget -O 'save.hdf5' 'https://www.dropbox.com/s/fbqhei3o9uocydl/save1.hdf5?dl=1'
python3 hw3_test.py $1 $2

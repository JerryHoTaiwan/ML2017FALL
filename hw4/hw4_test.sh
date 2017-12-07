#!/bin/bash
wget -O 'test5.h5' 'https://www.dropbox.com/s/vj0bs7p7wgxc1e0/test5.h5?dl=1'
wget -O 'test6_2.h5' 'https://www.dropbox.com/s/bu4kdycatwz947y/test6_2.h5?dl=1'
wget -O 'embed_model_128' 'https://www.dropbox.com/s/dj1em0uzlx7k8y6/embed_model_128?dl=1'
python3 hw4_test.py $1 $2

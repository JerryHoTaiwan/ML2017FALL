#!/bin/bash
wget -O 'embed_model_128' 'https://www.dropbox.com/s/dj1em0uzlx7k8y6/embed_model_128?dl=1'
python3 hw4_train.py $1 $2

#! /bin/bash

#%% This file runs the sweep.py file
#%% The sweep.py file takes a cmd line argument 'dataset_name' and runs a hyperparameter search over it


for name in "squares" "light"
do 
    python sweep.py --dataset-name=$name --test
done

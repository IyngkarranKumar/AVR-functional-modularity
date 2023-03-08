#!/bin/bash

#set up directory structure using python file

#TO RUN: bash gen.sh

#SET THESE FOR DSET GENERATION

parent_store_dir="big/"
n=1000




source_dir=" "
dest_dir="I-RAVEN/"
new_file="custom_const.py"
old_file="const.py"

python dir_edit.py --dest-dir $dest_dir --new-file $new_file --old-file $old_file --parent-store-dir=$parent_store_dir


#create three directories

list=("train test val")
for val in $list; do

    if [ $val == "train" ]
        then let num=($n*7)/10
    elif [ $val == "test" ]
        then let num=($n*3)/10
    else
        let num=($n*0)/10
    fi
    
    path=$parent_store_dir$val;
    python I-RAVEN/main.py --num-samples $num --save-dir $path
done


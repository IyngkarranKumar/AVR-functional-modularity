#!/bin/bash

#set up directory structure using python file

#bash gen.sh

#SET THESE FOR DSET GENERATION
source_dir=" "
dest_dir="I-RAVEN/"
new_file="custom_const.py"
old_file="const.py"

parent_store_dir="triangles/"
n=10

python dir_edit.py --dest-dir $dest_dir --new-file $new_file --old-file $old_file --parent-store-dir=$parent_store_dir


#create three directories

list=("train test val")
for val in $list; do

    if [ $val == "train" ]
        then let num=($n*7)/10
    elif [ $val == "test" ]
        then let num=($n*2)/10
    else
        let num=($n*1)/10
    fi
    
    path=$parent_store_dir$val;
    python I-RAVEN/main.py --num-samples $num --save-dir $path
done


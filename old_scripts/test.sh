#!/bin/bash
#NO LONGER REQD

x="train"


n=1000

if [ $x == "train" ]
    then 
        echo "train"
        let num=($n*7)/10
elif [ $x == "test" ]
    then 
        echo "test"
        let num=$n/4
else
    echo "val"
    let num=$n/11
fi

echo $num




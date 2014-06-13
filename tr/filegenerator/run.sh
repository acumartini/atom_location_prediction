#!/usr/bin/bash

dir="/home/users/wwe/prog/data/"
outdir="/home/users/wwe/prog/data_noise/"

for i in $(ls /home/users/wwe/prog/data/)
do
    echo $i
    export FIRST=$dir$i
    export SECOND=$outdir$i
    qsub -V -q batch -e err/$i.err -o log/$i.log sub.pbs

done

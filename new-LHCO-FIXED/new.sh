#!/bin/bash




echo $1 $2 $3 $4 $5 $6 $7
python3 testing-script.py "$1" "$2" "$3" "$4" "$5" "$6" "$7"
cd results
fn=$(ls -tc *.png | head -n1) ; eog $fn



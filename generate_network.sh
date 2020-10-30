#!/bin/bash



mkdir results
#echo $1 $2 $3 $4 $5 $6 $7
#python3 testing-script.py "$1" "$2" "$3" "$4" "$5" "$6" "$7"
python3 scripts/aifriendly.py 50 50 32 120 32 0 0
python3 scripts/aifriendly.py 50 50 32 120 32 spherical 0
python3 scripts/aifriendly.py 50 50 32 120 32 cartesian 0
python3 scripts/aifriendly.py 50 50 32 120 32 cartesian 1
#cd results
#fn=$(ls -tc *.png | head -n1) ; eog $fn



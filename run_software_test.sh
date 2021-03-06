#!/usr/bin/bash

time=$(date +"%Y-%m-%d--%H-%M-%S")

rundir="software_test_${time}"

cp -avr input $rundir

cd $rundir

# run the tests


mkdir new_images
cp -r images/* new_images/

python3 beta_den.py -p  -i new_images 
python3 beta_den.py -p -s -i new_images 


python3 beta_den.py -p -o > output_p_only

python3 beta_den.py -a -o > output_a_only
python3 beta_den.py -e -o > output_a_only
python3 beta_den.py -c -o > output_a_only


cd - 

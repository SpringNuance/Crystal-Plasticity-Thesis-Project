#!/bin/bash

run_until=$1
start_at=$2
fullpath=$3
material=$4

if SIM=$(sbatch --parsable --array=$start_at-$run_until array_subFFTCSC_heavy.sh $fullpath $material)
then
   sbatch -W --dependency=afterok:$SIM --array=$start_at-$run_until array_subFFTCSCpost_heavy.sh $fullpath $material
   exit 0
else
   exit 1
fi
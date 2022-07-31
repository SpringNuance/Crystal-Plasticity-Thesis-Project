#!/bin/bash

run_until=$1
start_at=$2

if SIM=$(sbatch --parsable --array=$start_at-$run_until array_subFFTCSC.sh)
then
   sbatch -W --dependency=afterok:$SIM --array=$start_at-$run_until array_subFFTCSCpost.sh 
   exit 0
else
   exit 1
fi
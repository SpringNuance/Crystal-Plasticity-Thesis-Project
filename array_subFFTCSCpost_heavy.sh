#!/bin/bash -l
# created: 13.07.2021
# author: WenqiLiu
#SBATCH --account=project_2004956
#SBATCH --partition=small
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH -J Array_CPPost
#SBATCH -e Array_CPPost
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

fullpath=$1
material=$2

echo ${SLURM_ARRAY_TASK_ID}

### Change to the work directory
cd ${fullpath}/${SLURM_ARRAY_TASK_ID}

ulimit -s unlimited

source /projappl/project_2004956/DAMASK/damask2.0.2/DAMASK/DAMASK_env.sh
PATH=$PATH:/projappl/project_2004956/DAMASK/damask2.0.2/DAMASK/processing/post
postResults.py --cr texture,f,p --time ${material}_tensionX.spectralOut
cd ${fullpath}/${SLURM_ARRAY_TASK_ID}/postProc


addCauchy.py ${material}_tensionX.txt
addStrainTensors.py --left --logarithmic ${material}_tensionX.txt
addMises.py -s Cauchy ${material}_tensionX.txt
addMises.py -e 'ln(V)' ${material}_tensionX.txt
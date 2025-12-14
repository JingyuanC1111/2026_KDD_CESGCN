#!/bin/bash

#SBATCH --job-name=MTE
#SBATCH --partition=bii
#SBATCH --account=nssac_students
#SBATCH --time=40:00:00
#SBATCH --mem=50000

module load miniforge
conda activate MTE
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python FLU_HOSP_MTE.py 200

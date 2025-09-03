#!/bin/bash -l
#
# Multithreading example job script for MPCDF Raven.
# In addition to the Python example shown here, the script
# is valid for any multi-threaded program, including
# Matlab, Mathematica, Julia, and similar cases.
#
#SBATCH --job-name=my_job_name
#SBATCH --output=KuroSiwo/logs/%j_out.log
#SBATCH --error=KuroSiwo/logs/%j_err.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000MB
#SBATCH --time=01:00:00
#SBATCH --reservation=workshop
# load necessary modules/ softwares
module load anaconda/3/2023.03
module load pytorch/gpu-cuda-12.1/2.2.0
# Set number of OMP threads to fit the number of available cpus, if applicable.
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun uv run ./KuroSiwo/main.py
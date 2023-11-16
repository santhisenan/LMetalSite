#!/bin/bash
# Exercise 1 submission script - submit.sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=24:mem=32G
#PBS -l walltime=48:00:00
#PBS -P personal-santhise
#PBS -N lmetalsite

# Commands start here
cd ${PBS_O_WORKDIR}
cd /home/users/ntu/santhise/code/LMetalSite
module load nvidia/23.7
# module load pytorch/1.11.0-py3-gpu
module load anaconda3/2022.10
conda activate lmetalsite
python -m script.extract
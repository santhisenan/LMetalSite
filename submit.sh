#!/bin/bash
# Exercise 1 submission script - submit.sh
# Below, is the queue
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:ncpus=24:mem=32G
#PBS -l walltime=00:10:00
#PBS -P personal-santhise
#PBS -N lmetalsite
# Commands start here
cd ${PBS_O_WORKDIR}
cd /home/users/ntu/santhise/code/LMetalSite

module load anaconda3/2022.10
conda activate lmetalsite
python -m script.extract
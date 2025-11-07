#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J cd_mult
#SBATCH --mail-user=nravi3@ur.rochester.edu
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00
#SBATCH -o /pscratch/sd/n/nravi/GV_classification/cd_calc_mult.out
#SBATCH -e /pscratch/sd/n/nravi/GV_classification/cd_calc_mult.err

source /global/common/software/desi/desi_environment.sh main

#run the application:
srun parallel --jobs 110 --link python /global/homes/n/nravi/GV_classification/color_gradient_main_multiprocess.py {1} {2} ::: {0..635579..5831} ::: {5831..641410..5831}
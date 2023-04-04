#! /bin/bash



#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --mem=4g
#SBATCH -p ug-gpu-small
#SBATCH --qos=debug
#SBATCH --job-name=SCL_70
#SBATCH -t 01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user qclx31@durham.ac.uk



/bin/bash sweep.sh
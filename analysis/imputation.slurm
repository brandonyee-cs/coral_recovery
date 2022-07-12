#!/bin/bash
#SBATCH --job-name=imp
#SBATCH --partition=shared
## 3 day max run time for public partitions, except 4 hour max runtime for the sandbox partition
#SBATCH --time=0-24:00:00 ## time format is DD-HH:MM:SS
#SBATCH --cpus-per-task=4
#SBATCH --mem=10000 ## max amount of memory per node you require
##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node

#SBATCH --error=hello-%A.err ## %A - filled with jobid
#SBATCH --output=hello-%A.out ## %A - filled with jobid

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=druryc@hawaii.edu

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html
source ~/.bash_profile

module load lang/Java
module load lang/Anaconda3
module load lang/R
source activate tools_seq

Rscript imputation.R &>>log_imputation.txt

#!/bin/bash
#SBATCH --job-name=clonality_GL
#SBATCH --partition=shared
#SBATCH --time=0-60:00:00 ## time format is DD-HH:MM:SS
#SBATCH --cpus-per-task=16
#SBATCH --mem=50000 ## max amount of memory per node you require
##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node
#SBATCH --error=hello-%A.err ## %A - filled with jobid
#SBATCH --output=hello-%A.out ## %A - filled with jobid
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=druryc@hawaii.edu
#srun -I30 -p sandbox -N 1 -c 1 --mem=6G -t 0-01:00:00 --pty /bin/bash

source ~/.bash_profile
#module load lang/R/3.5.1-foss-2018b
module load lang/Python/3.7.2-intel-2018.5.274
module load bio/Biopython/1.73-intel-2018.5.274-Python-2.7.15
module load lang/SciPy-bundle/2019.03-intel-2018.5.274
#module load bio/BLAST+/2.2.31
module load lang/Anaconda3/5.1.0
source activate tools_seq
#source activate ngsLD
#export LD_LIBRARY_PATH=/home/druryc/.conda/envs/ngsLD/lib

#angsd -bam bamlist.txt -nThreads 16 -ref ~/reference/mcap.fasta -doMajorMinor 4 -GL 2 -doMaf 2 -doCounts 1 -dumpCounts 2 -doPost 1 -doGeno 8 -doGlf 2 -SNP_pval 2e-6 -minInd 200 -minmapQ 30 -minQ 20 -geno_minDepth 2 -out output
angsd -bam bamlist.txt -nThreads 16 -ref ~/reference/ragtag.scaffolds.fasta -doCounts 1 -doIBS 1 -makeMatrix 1 -minInd 10 -minmapQ 30 -minQ 30 -out mcap_relatedness_f
#!/bin/bash
#
#SBATCH --job-name=Exemple
#SBATCH --comment=35985002F
#SBATCH --ntasks=96
##SBATCH --time=00:30:00
#SBATCH --signal=B:SIGUSR1@60
#SBATCH --output=EXPLE.%J.out
#SBATCH --error=EXPLE.%J.err
#SBATCH --exclusive
#SBATCH --requeue 
##SBATCH --dependency=99999
##SBATCH --constraint="[skl|csl]"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=baptiste.romain@onera.fr

#-----------------------------------------------------------------------------

# DESCRIPTION

#-----------------------------------------------------------------------------

# ADDITIONNAL PARAMETERS


#-----------------------------------------------------------------------------

# ENVIRONMENT
echo 'ENVIRONMENT='
module purge
conda activate envNeuNeu

#-----------------------------------------------------------------------------

# INFOS
echo 'CLUSTER =' $SLURM_CLUSTER_NAME
echo 'NPROCS  =' $SLURM_NTASKS
echo 'PART    =' $SLURM_JOB_PARTITION
echo 'QOS     =' $SLURM_JOB_QOS
echo 'JOBNAME =' $SLURM_JOB_NAME

sleep 1

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------

python ISOC_db_gen.py

#-----------------------------------------------------------------------------

# PP

#-----------------------------------------------------------------------------

# CLEANING



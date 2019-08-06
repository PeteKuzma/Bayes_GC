#!/bin/bash
# Grid Engine options (lines prefixed with #$)
#$ -N omega_attemptback
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l h_vmem=4G
#  These options are:
#  job name: -N
#  use the current worcalking directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
#$ -m beas
#$ -M pkuzma@roe.ac.uk
. /etc/profile.d/modules.sh
module load anaconda
module load openmpi
source activate PyMN
export LD_LIBRARY_PATH=$HOME/multinest/MultiNest/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/own_codes:$PYTHONPATH
#mpirun -n 16 python omega_calking.py
#mpirun -n 32 python omega_calking2.py
#mpirun -n 32 python omega_calk52.py
mpirun -n 32 python omega_calk_orig.py
mpirun -n 32 python omega_calk5.py
#mpirun -n 16 python omega_calk5.py
#mpirun -n 16 python omega_calking.py

#python omega_calk5_orig.py
#python omega_calk_orig.py
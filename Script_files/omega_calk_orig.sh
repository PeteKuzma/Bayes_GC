#!/bin/bash
# Grid Engine options (lines prefixed with #$)
#$ -N omega_attemptback  
#$ -cwd                  
#$ -l h_rt=48:00:00 
#$ -l h_vmem=14G
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

#python N7089_calking_iso.py
#python N7089_calking.py
#python N7089_calkingoldPM.py

#python Pal5_calking.py
#python Pal5_calkingoldPM.py
python Pal5_calking_iso.py

#python omegaORIG_calking_iso.py
#python omegaORIG_calkingoldPM.py

#python omega_calk5_orig.p
#python omega_calk_orig.py

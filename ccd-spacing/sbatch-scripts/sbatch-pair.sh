#!/bin/bash

##SBATCH --partition=rubin
#SBATCH --job-name=calib1
#SBATCH --output=/sdf/group/rubin/user/esteves/github/slacSummer2023/ccd-spacing/log/out-pair.txt
#SBATCH --error=/sdf/group/rubin/user/esteves/github/slacSummer2023/ccd-spacing/log/err-pair.txt
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=1:00:00

source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2023_22/loadLSST-ext.bash
setup lsst_distrib

export PYTHONPATH="/sdf/group/rubin/user/esteves/github/mixcoatl/python:${PYTHONPATH}" # Needed for using gridFit
 
# Set up repositories for collections
export REPO=/sdf/group/rubin/repo/main/
export CONFIG=/sdf/group/rubin/user/esteves/github/slacSummer2023/ccd-spacing/

echo "Hey I started"
date

## Run tasks
pipetask run \
        -j 8 \
        -d "instrument='LSSTCam' AND exposure.observation_type='spot' AND exposure.science_program IN ('13227', '13229') AND detector in (31, 32)" \
        -b ${REPO} \
        -i LSSTCam/calib,LSSTCam/raw/all,u/jesteves/2023.06.12/det32Pair \
        -o u/jesteves/2023.06.15/pair32 \
        -p ${CONFIG}/spots.yaml \
        --skip-existing \
        --extend-run \
        --clobber-outputs \
        --register-dataset-types 
        
echo "Hey I finished"
date

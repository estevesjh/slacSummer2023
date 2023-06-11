#!/bin/bash

##SBATCH --partition=rubin
#SBATCH --job-name=calib1
#SBATCH --output=/sdf/group/rubin/user/esteves/github/slacSummer2023/ccd-spacing/out.txt
#SBATCH --error=/sdf/group/rubin/user/esteves/github/slacSummer2023/ccd-spacing/err.txt
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=16G
#SBATCH --time=5:00:00

source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2023_22/loadLSST-ext.bash
setup lsst_distrib

export PYTHONPATH="/sdf/group/rubin/user/esteves/github/mixcoatl/python:${PYTHONPATH}" # Needed for using gridFit
 
# Set up repositories for collections
export REPO=/sdf/group/rubin/repo/main/
export CONFIG=/sdf/group/rubin/user/esteves/github/slacSummer2023/ccd-spacing/
export SBIAS=u/cslage/calib/13144/bias_20211229
export SDARK=u/cslage/calib/13144/dark_20211229
export DEFECTS=u/cslage/calib/13144/defects_20211229


## Run tasks
pipetask run \
        -j 6 \
        -d "instrument='LSSTCam' AND exposure.observation_type='spot' AND detector not in (194, 196, 201, 202, 203, 204) AND exposure.science_program IN ('13228', '13229')" \
        -b ${REPO} \
        -i LSSTCam/calib,LSSTCam/raw/all \
        -o u/jesteves/2023.06.09/spots.2023.06.09b \
        -p ${CONFIG}/spots.yaml \
        --register-dataset-types
        
echo "Hey I finished"
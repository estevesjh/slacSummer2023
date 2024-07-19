#!/bin/bash

##SBATCH --partition=rubin
#SBATCH --job-name=calib1
#SBATCH --output=/sdf/group/rubin/user/esteves/github/slacSummer2023/ccd-spacing/log/all-out.txt
#SBATCH --error=/sdf/group/rubin/user/esteves/github/slacSummer2023/ccd-spacing/log/all-err.txt
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00

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
        -j 16 \
        -d "instrument='LSSTCam' AND exposure.observation_type='spot' AND exposure.science_program IN ('13227', '13229') AND detector not in (189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205)" \
        -b ${REPO} \
        -i LSSTCam/calib,LSSTCam/raw/all \
        -o u/jesteves/2023.06.12/test_b \
        -p ${CONFIG}/spots.yaml \
        --skip-existing \
        --extend-run \
        --clobber-outputs \
        --register-dataset-types 
        
echo "Hey I finished"

import pandas as pd
import numpy as np
from utils import *

##################################################################
########################## SETUP #################################
###### Infile Setup
path = '../data/ccsTemp/'
nameRaft = path+'temp%s-Run6.csv'
e2vRaftList = ["R11", "R12", "R13", "R14", "R21", "R22", 
               "R23", "R24", "R30", "R31", "R32", "R33", "R34"]

###### Ouftile
fnameRaft = path+'temp%s-Run6-dynamic.csv'
fnameBadRaft = path+'bad-sensors-dynamic-%s.csv'

####### Time Cuts
end_date = '2023-06-24 20:00:00'
start_date = '2023-06-23 00:00:0'

# average over time windows
# first df sampling
dt_group_sample = '10s'

# used only to find the groups (bad/good)
dt_roubst_mean_sample = '5min'

# used at the final sampling of the df
dt_final_sampling = '5min'
##################################################################

def evalRaft(raft):
    ## Load File
    df = read_file(nameRaft%raft, start_date, end_date)

    # Group the sensor temp
    # Temp_R33-S00, Temp_R33-S01, ...
    dfn = group_temps(df, dt_group_sample)

    # Define the Bad Sensors
    # Use a Kmeans algorithm to find two groups
    # Take the mean over the good group
    # Define a new column to the offsets
    dfn2, bad_sensors = get_robust_mean(dfn, dt_roubst_mean_sample)
    dfOut = dfn2.resample(dt_final_sampling).mean().interpolate()

    # Print a repport
    mydict = to_dict_bad_sensors_report(dfOut, bad_sensors)
    bad_sensors_repport(mydict)
    write_dict_to_csv(fnameBadRaft%raft, mydict)

    # save the file
    fname = fnameRaft%raft
    dfOut.to_csv(fname)

if __name__ == "__main__":
    for raft in e2vRaftList:
        evalRaft(raft)
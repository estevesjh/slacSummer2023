import csv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

path = '../data/ccsTemp/'
nameRaft = path+'temp%s-Run6.csv'

def read_file(fname, start_date, end_date):
    # fast reading
    dt1 = np.dtype({'names': ['time', 'temp', 'channel','datetime'],
                'formats': [int , float, 'U28', 'U28']})
    rdata = np.loadtxt(fname, dt1, usecols=(0, 1, 3, 4))
    
    # convert to dataframe
    df = pd.DataFrame(rdata)
    
    # convert datetime to a time-series
    df['datetime'] = pd.to_datetime(df['datetime'])#.dt.strftime('%d-%m %H:%M:%S')
    df = df.set_index('datetime')
    # make sensor, raft-sensor columns
    df['raft-sensor'] = df['channel'].apply(lambda x: '-'.join([x.split('/')[1],x.split('/')[3]]))
    df['sensor'] = df['channel'].apply(lambda x: x.split('/')[3])
    df = df.drop('channel',axis=1)
    df = df.drop('time',axis=1)
    
    mask = (df.index >= start_date) & (df.index <= end_date)
    dfmask = df.loc[mask]
    
    return dfmask.sort_index()

def group_temps(df,tsample='10s'):
    df_pivot = df.pivot_table(index=df.index, columns='raft-sensor', values='temp', aggfunc='mean')
    df_pivot.columns = ['Temp_' + col for col in df_pivot.columns]
    df_merged = df.merge(df_pivot, left_index=True, right_index=True)
    df_merged = df_merged.resample(tsample).mean().interpolate()
    df_merged = df_merged.drop('temp',axis=1)
    return df_merged

def get_groups(df, ngroups=2):
    x = df.to_numpy().T
    kmeans = KMeans(n_clusters=ngroups, random_state=0).fit(x)
    labels = kmeans.labels_
    return labels

def create_dT_cols(df, tmean, tstd):
    columns_to_subtract = [col for col in df.columns if 'Temp_R' in col]

    for col in columns_to_subtract:
        df[r'Delta_'+col] = df[col] - tmean
        df[r'Delta_STD_'+col] = (df[col] - tmean)/tstd
    return df

def to_dict_bad_sensors_report(df, bad_sensors):
    raft = bad_sensors[0].split('-')[0]
    tmean, tstd = df['T_%s_mean'%raft].mean(), df['T_%s_std'%raft].mean()
    columns = ['Delta_Temp_%s'%col for col in bad_sensors]
    offsets = [df[col].mean() for col in columns]
    out = {'raft':raft, 'sensor-list':bad_sensors, 
           'tmean':tmean, 'tstd':tstd, 'offset-list':offsets}
    return out

def write_dict_to_csv(filename, out):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Raft', 'Sensor', 'Tmean', 'Tstd', 'Offset'])
        for i in range(len(out['sensor-list'])):
            writer.writerow([out['raft'], out['sensor-list'][i], out['tmean'], out['tstd'], out['offset-list'][i]])
    #print('Saved file: %s'%filename)

def bad_sensors_repport(mydict):
    str_offsets = ', '.join('%.3f'%num for num in mydict['offset-list']) 
    print(6*'-----')
    print('%s Sensors RTD Temp Summary'%mydict['raft'])
    print('Good Sensors [C]')
    print('mean, std: %.3f, %.3f\n'%(mydict['tmean'], mydict['tstd']))
    print('Bad Sensors')
    print('sensor-name : '+', '.join(mydict['sensor-list']))
    print('Temp. Offset: '+ str_offsets)
    print(6*'-----')

def pick(df, labels):
    nk = np.unique(labels).size
    nsize = [len(df.columns[labels==i]) for i in range(nk)]
    good = list(df.columns[labels == np.argmax(nsize)])
    bad = list(df.columns[labels != np.argmax(nsize)])
    return good, bad

def get_robust_mean(df, tsample='5min'):
    # raft name
    raft = list(df.columns)[0].split('_')[1][:3]
    
    # interpolate nan values
    df_int = df.resample(tsample).mean().interpolate()
    
    # group bw bad and good
    labels = get_groups(df_int)
    
    # pick bad or good
    good_sensors, bad_sensors = pick(df, labels)

    # take the mean, std of the good sensors
    df['T_%s_mean'%raft] = df[good_sensors].mean(axis=1)
    df['T_%s_std'%raft] = df[good_sensors].std(axis=1)
    
    # create delta T columns with subtraction over the mean
    df = create_dT_cols(df, df['T_%s_mean'%raft], df['T_%s_std'%raft])
    
    # put only RXX-SXX
    bad_RXXSXX = [si.split('_')[1] for si in bad_sensors]
    return df, bad_RXXSXX
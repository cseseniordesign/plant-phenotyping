import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import scipy.misc
import sys
#sys.path.append('/home/schnablelab/cmiao/MyRepo/schnablelab/apps')
from schnablelab.apps.DirDataframe import GenDirDataframe
import os

check_path = Path('/work/schnablelab/cmiao/TimeSeriesGWAS/High_throughput_Phenotyping/Experiment2/HyperImages/CheckZoomLevel')
df = GenDirDataframe(check_path, '*npy')
df['sm'] = df['fn'].apply(lambda x: x.split('_')[0])
df['date'] = df['fn'].apply(lambda x: x.split('_')[-1].split('.')[0])
#df['date'] = pd.to_datetime(df['date'])
df1 = df.sort_values(['date','sm']).reset_index(drop=True)
counts = df1['date'].value_counts().sort_index()

def checkZoom(date):
    os.mkdir(check_path/date)
    df_sd = df1[df1['date']==date]
    for fn in df_sd['fnpath']:
        sm = fn.name.split('_')[0]
        print(sm)
        npy = np.load(fn)
        npy20 = npy[:,:,20]
        #plt.imshow(npy20, cmap='gray')
        img_fn = '%s.png'%sm
        scipy.misc.imsave(check_path/date/img_fn, npy20)

if len(sys.argv)==2:
    checkZoom(sys.argv[1])
else:
    print('date')


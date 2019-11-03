from schnablelab.apps.header import Slurm_header
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

for i in counts.index.tolist():
    cmd = 'python /home/schnablelab/cmiao/MyRepo/schnablelab/CNN/checkZoom.py %s'%i
    print(cmd)
    header = Slurm_header%(2, 10000, i, i, i)
    header += "ml anaconda\nsource activate MCY\n"
    header += cmd
    with open('%s.slurm'%i, 'w') as f:
        f.write(header)
        print('%s.slurm has been generated.'%i)

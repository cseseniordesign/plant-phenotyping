#!/usr/bin/env python

import os
import pwd
import sys
import time
from glob import glob 
from Pegasus.DAX3 import *

# The name of the DAX file is the first argument
if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s DAXFILE\n" % (sys.argv[0]))
        sys.exit(1)
daxfile = sys.argv[1]

USER = pwd.getpwuid(os.getuid())[0]

# Create a abstract dag
dax = ADAG("split")

# Add some workflow-level metadata
dax.metadata("creator", "%s@%s" % (USER, os.uname()[1]))
dax.metadata("created", time.ctime())

path_list = glob("/work/csesd/pnnguyen/data/*/Hyp_SV_90/") 

for path in path_list:
	corn_folder_name = path.split('/')[5]
	corn_folder_name = corn_folder_name.replace(' ','_')
	batch_preprocess = Job("python3")
	batch_preprocess.addArguments("/work/csesd/johnsuzh/pics2predictions/schnablelab/CNN/Preprocess.py","hyp2arr",path,corn_folder_name)
	dax.addJob(batch_preprocess)
	nparr = File("%s.npy" % corn_folder_name)
	batch_preprocess.uses(nparr, link=Link.OUTPUT, transfer=False, register=False)
	prediction = File("model_4_300_3.%s.prd.png" % corn_folder_name)
	predict = Job("python3")
	model = File("model_4_300_3.40421104694053e-05.h5")
	predict.addArguments("/work/csesd/johnsuzh/pics2predictions/schnablelab/CNN/Predict_snn.py","Predict",model,nparr)
	predict.uses(model, link=Link.INPUT)
	predict.uses(nparr, link=Link.INPUT)
	predict.setStdout(prediction)
	predict.uses(prediction, link=Link.OUTPUT, transfer=True, register=True)
	dax.addJob(predict)
	
	dax.depends(predict, batch_preprocess)

f = open(daxfile, "w")
dax.writeXML(f)
f.close()

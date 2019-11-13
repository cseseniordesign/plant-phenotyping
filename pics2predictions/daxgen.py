#!/usr/bin/env python

import os
import pwd
import sys
import time
import file_paths_config as paths
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

path_list = glob(paths.file_paths['data']) 

for path in path_list:
	corn_folder_name = path.split('/')[paths.file_paths['data_index']]
	corn_folder_name = corn_folder_name.replace(' ','_')
	preprocess = Job("python3")
	preprocess.addArguments("-m", "schnablelab.CNN.Preprocess","hyp2arr",path,corn_folder_name)
	dax.addJob(preprocess)
	nparr = File("%s.npy" % corn_folder_name)
	preprocess.uses(nparr, link=Link.OUTPUT, transfer=False, register=False)
	prediction = File("model_4_300_3.%s.prd.png" % corn_folder_name)
	predict = Job("python3")
	model = File(paths.file_paths['model'])
	predict.addArguments("-m", "schnablelab.CNN.Predict_snn","Predict",model,nparr)
	predict.uses(model, link=Link.INPUT)
	predict.uses(nparr, link=Link.INPUT)
	predict.setStdout(prediction)
	predict.uses(prediction, link=Link.OUTPUT, transfer=True, register=True)
	dax.addJob(predict)
	dax.depends(predict, preprocess)

f = open(daxfile, "w")
dax.writeXML(f)
f.close()

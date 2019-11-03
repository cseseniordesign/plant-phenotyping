#!/usr/bin/env python

import os
import pwd
import sys
import time
from Pegasus.DAX3 import *

# The name of the DAX file is the first argument
if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s DAXFILE\n" % (sys.argv[0]))
        sys.exit(1)
daxfile = sys.argv[1]

USER = pwd.getpwuid(os.getuid())[0]

dax = ADAG("pipeline")

# Add some workflow-level metadata
dax.metadata("creator", "%s@%s" % (USER, os.uname()[1]))
dax.metadata("created", time.ctime())

nparr = File("test.npy")

preprocess = Job("python3")
preprocess.addArguments("/work/csesd/johnsuzh/pic2prediction/input/schnablelab/CNN/Preprocess.py","hyp2arr","/work/csesd/johnsuzh/pic2prediction/input/Hyp_SV_90","test")
preprocess.uses(nparr, link=Link.OUTPUT)
dax.addJob(preprocess)

prediction = File("model_4_300_3.test.prd.png")
model = File("model_4_300_3.40421104694053e-05.h5")
predict = Job("python3")
predict.addArguments("/work/csesd/johnsuzh/pic2prediction/input/schnablelab/CNN/Predict_snn.py","Predict",model,nparr)
predict.uses(model, link=Link.INPUT)
predict.uses(nparr, link=Link.INPUT)
predict.setStdout(prediction)
predict.uses(prediction, link=Link.OUTPUT, transfer=True, register=False)
dax.addJob(predict)

#dax.depends(predict, preprocess)

f = open(daxfile, "w")
dax.writeXML(f)
f.close()

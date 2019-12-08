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

plant_ids = set()
path_list = glob(paths.file_paths['data'])
path_list_index = paths.file_paths['data'].split('/').index('*')
current_dir = os.getcwd()
output_dir = current_dir + "/output"
plant_phenotyping_index = current_dir.split('/').index('plant-phenotyping')
plant_phenotyping_path = "/".join(current_dir.split('/')[:plant_phenotyping_index + 1])
plant_predict_jobs =  {}
plant_extract_jobs = {}

for path in sorted(path_list):
	plant_folder_name = path.split('/')[path_list_index]
	model = paths.file_paths['model']
	model_str = model.split('.')[0]
	numpy_name = plant_folder_name.split('.')[0]
	plant_name = numpy_name.split('_')[0]
	prediction = File("%s.%s.prd.png" % (model_str, numpy_name))
	predict = Job("python3")
	model_file = File(model)
	predict.addArguments("-m", "schnablelab.CNN.Predict_snn","Predict", model_file, path)
	predict.uses(model, link=Link.INPUT)
	predict.setStdout(prediction)
	predict.uses(prediction, link=Link.OUTPUT, transfer=True, register=True)
	dax.addJob(predict)
	if plant_name not in plant_ids:
		plant_ids.add(plant_name)
		measure = Job("python3")
		measure.addArguments(plant_phenotyping_path + "/traits_extraction.py", "-i", plant_name, "-p",output_dir)
		csv_name = "plant_traits_" + plant_name + ".csv"
		csv = File(csv_name)
		measure.uses(prediction, link=Link.INPUT)
		measure.uses(csv, link=Link.OUTPUT, transfer=True, register=False)
		measure.setStdout(csv)
		plant_extract_jobs[plant_name] = measure
		plant_predict_jobs[plant_name] = [predict]
	else:
		plant_predict_jobs[plant_name].append(predict)
		measure.uses(prediction, link=Link.INPUT)

#print(plant_extract_jobs)
for extract_job in plant_extract_jobs:
	job = plant_extract_jobs[extract_job]
	dax.addJob(job)
	for predict_job in plant_predict_jobs[extract_job]:
		dax.depends(child=job, parent=predict_job)
		

f = open(daxfile, "w")
dax.writeXML(f)
f.close()

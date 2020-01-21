__author__ = 'plant-phenotyping'

from zipfile import ZipFile
import os
import argparse

def unzip(plant_ID, date, image_type):
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-n", "--name", required=False, help="plant ID")
    # ap.add_argument("-d", "--date", required=False, help="date")
    # ap.add_argument("-t", "--type", required=False, help="image type")
    #ap.add_argument("-f", "--files", nargs="*",  required=False, help="predict files")
    # args = vars(ap.parse_args())
    # plant_ID = args["name"]
    # date = args["date"]
    # image_type = args["type"]


    path = "output/"
    files = os.listdir(path)
    for file in files:
        if (plant_ID in file) and (date in file):
            file_name = file

    folder_name = file_name[0:-4]

    with ZipFile(path + file_name, 'r') as zip:
        for file in zip.namelist():
            if file.startswith(folder_name+'/'+image_type):
                zip.extract(file)
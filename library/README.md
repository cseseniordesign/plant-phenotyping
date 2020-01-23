# Library Documentation
# SD2019-Plant Phenotyping
## Dependencies
* numpy
* opencv

## Functionality
1. info(plant_ID, date)
* input: plant_ID, date
* output: print types of images that are available
* example: ```tools.info("JS39-65", "2018-04-11")```
2. unzip(plant_ID, date, image_type):
* input: plant_ID, date, image_type
* output: the folder of images that match plant ID, date, and image type.
* example: ```tools.unzip("JS39-65", "2018-04-11", "Nir")```
3. preprocess(plant_ID, date):
* input: plant_ID, date
* output: numpy arrays of Hyperspectral images
* example: ```tools.preprocess("JS39-65", "2018-04-11")```




## Running the library
1. Warnings
* the plant_ID should be in a format like "JS39-65", the date should be in a format like "2018-04-11"
* the possible image types are Hyp, Nir, Vis, Fluo, IR
* Hyperspectral images should be reconstructed first, before running the "preprocess" to produce the numpy array
2. import the module as a Python package
* `from library import tools`
* `tools.info([plant_ID], [date])` 
* `tools.unzip([plant name], [date], [image type])`
* `tools.preprocess([plant name], [date])`
3. running the module in terminal 
* `python3 -m library.tools info -n JS39-65 -d 2018-04-11`
* `python3 -m library.tools unzip -n JS39-65 -d 2018-04-11 -t Hyp`
* `python3 -m library.tools preprocess -n JS39-65 -d 2018-04-11`

## Demonstration
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/library_demo.png)






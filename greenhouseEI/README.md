# GreenhouseEI Documentation
# SD2019-Plant Phenotyping
## Dependencies
* numpy
* opencv

## Functionality
1. info(plant_ID, date, path)
* input: plant_ID, date, path
* output: print types of images that are available
* example: ```tools.info("JS39-65", "2018-04-11", "/Users/john/PycharmProjects/Library_SD/output")```
2. unzip(plant_ID, date, image_type, path):
* input: plant_ID, date, image_type, path
* output: the folder of images that match plant ID, date, and image type.
* example: ```tools.unzip("JS39-65", "2018-04-11", "Nir", "/Users/john/PycharmProjects/Library_SD/output")```
3. preprocess(plant_ID, date, path):
* input: plant_ID, date, path
* output: numpy arrays of Hyperspectral images
* example: ```tools.preprocess("JS39-65", "2018-04-11", "/Users/john/PycharmProjects/Library_SD/output")```




## Running the library
1. Warnings
* the plant_ID should be in a format like "JS39-65", the date should be in a format like "2018-04-11"
* the possible image types are Hyp, Nir, Vis, Fluo, IR
* Hyperspectral images should be reconstructed first, before running the "preprocess" to produce the numpy array
2. import the module as a Python package
* `from greenhouseEI import tools`
* `tools.info([plant_ID], [date])` 
* `tools.unzip([plant name], [date], [image type])`
* `tools.preprocess([plant name], [date])`
3. running the module in terminal 
* `python3 -m greenhouseEI.tools info -n JS39-65 -d 2018-04-11 -p /Users/john/PycharmProjects/Library_SD/output`
* `python3 -m greenhouseEI.tools unzip -n JS39-65 -d 2018-04-11 -t Hyp -p /Users/john/PycharmProjects/Library_SD/output`
* `python3 -m greenhouseEI.tools preprocess -n JS39-65 -d 2018-04-11 -p /Users/john/PycharmProjects/Library_SD`

## Demonstration
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/library_demo.png)






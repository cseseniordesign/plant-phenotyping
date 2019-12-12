# How to use Pegasus on HCC server
# SD2019-Plant Phenotyping
## Dependencies
* pegasus (does not need to be installed)
* pandas
* matplotlib
* scikit-learn
* pillow
* opencv
* tensorflow
* keras
* numpy
* imutils
* scipy (version 1.1.0)

## Cloning Project (Need to do only once)
1. You can clone the project using: `git clone https://github.com/cseseniordesign/plant-phenotyping.git`
2. Run this command in the location you wish to clone the project to.

## Transferring Data to HCC Server (Need to do only once)
1. This workflow can only be run on the HCC server, so ensure the entire project (plant-phenotyping) is in your work directory on the HCC server.
2. Ensure that the schnablelab project (<https://github.com/freemao/schnablelab>) is on in the HCC server and is accessible by you.
    1. Run this command, `git clone https://github.com/freemao/schnablelab.git`, in the desired location to add the HCC server.
3. Ensure that the plant dataset is on the HCC server and is accessible by you.
    * Data transfer instructions are below for moving local information to the HCC servers:
        * Mac/Linux (under File Transferring With HCC Supercomputers):<https://hcc.unl.edu/docs/connecting/for_maclinux_users/>
        * Windows (under File Transferring With HCC Supercomputers): <https://hcc.unl.edu/docs/connecting/for_windows_users/>

## Creating Anaconda Environment to Run Workflow (Need to do only once)
1. Install dependencies: [pandas, matplotlib, scikit-learn, pillow, opencv, tensorflow, keras, and numpy, imutils, scipy (version 1.1.0)]
    1. Load anaconda: `module load anaconda`
    2. Create your own environment: `conda create -n [env_name]` (Where [env_name] is the name that you want your environment to be)
    3. Install dependencies [pandas, matplotlib, scikit-learn, pillow, opencv, tensorflow, keras, numpy, imutils, scipy (version 1.1.0)]: `conda install -n [env_name] [package_name]`
      * Example (normal): `conda install -n sd pandas`
      * Example (specify version): `conda install -n sd scipy==1.1.0`

## Modifying Workflow Paths to Work with File Structure (Need to do only once)
1. Modify file\_paths\_config.py located in the pics2predictions folder:
    * Change the data value to the file path to the dataset’s Hyp\_SV\_90 folders or the dataset's numpy files.
    * Change the model value to the value of the model you want use to predict the images (also ensure that the file is in the input folder).
    * Example (hyperspectral images):
    ` 'data': '/work/csesd/pnnguyen/plant-phenotyping/pics2predictions/test_data_hyp/*/Hyp_SV_90/',`
    ` 'model': 'fold3_model_4_300_0.0005.h5'`
    * Example (numpy arrays):
    ` 'data': '/work/csesd/pnnguyen/plant-phenotyping/pics2predictions/test_data_npy/*',`
    ` 'model': 'fold3_model_4_300_0.0005.h5'`
    * data needs to be in this structure for workflow to work for hyperspectral images:
        * data (does not have to be called data)
            * [plant folder name]
                * Hyp\_SV\_90
            * [plant folder name]
                * Hyp\_SV\_90
            * ....
    * Note: There must be just plant folders in the data folder and each plant folder must contain a Hyp\_SV\_90 folder.
    * data needs to be in this structure for daxgen_npy workflow to work:
      * data (does not have to be called data)
        * [numpy name].npy
        * [numpy name].npy
        * ...

2. Modify run_python.sh located in the pics2predictions folder:
    1. Change conda activate numpy to `conda activate [path to environment]` (where [path to environment] is the path of the environment you created).
        * Most anaconda paths on the HCC server are either located in $HOME or $WORK. You can check by by typing the command `ls -a` on the command line in either directories and checking if a .conda folder is in the directory.
       * The actual path to your enviroment is of the following pattern. $HOME/.conda/envs/[name of environment] or $WORK/.conda/envs/[name of environment]. ex:
    `conda activate /home/csesd/pnnguyen/.conda/envs/sd`

    2. Change the python path (content after export PYTHONPATH=) to the path to the schnablelab project. ex: (Note schnablelab folder is in the run folder in this example)
`export PYTHONPATH=/work/csesd/pnnguyen/run:$PYTHONPATH`

## Changing Shell Scripts to Executables (Need to do only once)
1. We will need to change the permissions for our scripts in the pics2predictions folder so we need to run the following commands in the pics2predictions folder:
    * `chmod +x generate_dax.sh` to make generate_dax.sh an executable.
    * `chmod +x plan_dax.sh` to make plan_dax.sh an executable.
    * `chmod +x run_python.sh` to make run_python.sh an executable.  

## Running Workflow
1. Run the command:
`./generate_dax.sh [dax file name].dax`
(Where [dax file name] is the name you want for the dax file to generate the dax file
2. Then the command :
`./plan_dax.sh [dax file name].dax`
to plan the dax and run the workflow
3. Use `pegasus-status -l [copy this from the output]` to see the status. (optional)
Example: `pegasus-status -l /work/csesd/johnsuzh/pics2predictions/submit/johnsuzh/pegasus/split/run0010`
This is to show the status of the workflow.
3. Use `pegasus-remove [copy this from the output]` to remove the current running workflow. (optional)
Example:`pegasus-remove /work/csesd/johnsuzh/pics2predictions/submit/johnsuzh/pegasus/split/run0010`
This is to remove the workflow from the queue.
4. Use `pegasus-analyzer [copy this from the output]` for a failing workflow to see the error message (optional)
Example:`pegasus-analyzer /work/csesd/johnsuzh/pics2predictions/submit/johnsuzh/pegasus/split/run0010`
This is to show the error message of the workflow if there was any error.
5. When it’s done, the status should be 100% in the %DONE column (use `pegasus-status` command). And you can see the output in the output folder.
6. The final outputs to the current workflow should be:
  * Prediction images (.png)
  * Trait extraction data (.csv)
  * Graph images that plotted points in .csv(s) (.png)
7. When you want to run the workflow again, make sure to remove the contents in the output folder.

## Example of using this Workflow
1. In this example, the plant-phenotyping and schnablelab repository are all on the HCC sever. Also, the anaconda environment has already been created with all of the dependencies of the project. Let us go step by step of using this workflow.
2. We are currently in the workflow directory, pics2predictions, you can see all of the files of the workflow directory with `ls`
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/view_workflow_directory.png)
3. Let us check the files that need to be configured to properly run the workflow. Here is the file_paths_config.py file that contains the paths to the dataset and the model that we want to use to predict the images. Here is shows the path to my test dataset and the model I want to use.
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/file_paths_config.png)
4. Let us now check run_python.sh. We see that PYTHONPATH is set to the directory that schnablelab is in. We se that it also activates my anaconda environment sd.
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/run_python.png)
5. We now need to convert all of the shell scripts into executables. We do this with `chmod +x [file name]` as shown below. The files are now in green after `ls`.
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/chmod.png)
6. Now we can generate the dax with `./generate_dax [dax file name].dax` we use test.dax as a test. This now shows test.dax in the directory.
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/generate_dax.png)
7. We can use plan_dax.sh to plan the dax file and run the workflow. Let us run `./plan_dax [dax file name].dax`.
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/plan_dax.png)
8. We can check the status with the line that was outputted from the plan dax command (`pegasus status -l ...`). The staging in jobs have just begun.
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/status_start.png)
9. After a few minutes with the test dataset, we can see that the workflow is done with (`pegasus status -l ...`).
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/status_done.png)
10. Let us go into our output directory and check the output with `ls`. We can see it outputted several predict images, one measurement .csv, and one growth .png.
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/output.png)
11. Here is an example of a predicted image.
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/predict.png)
12. Here is an example of the measurement .csv.
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/measure.png)
13. Here is an example of the growth .png.
![](https://github.com/cseseniordesign/plant-phenotyping/blob/master/illustrations/growth.png)
14. This concludes running the example of running this workflow.

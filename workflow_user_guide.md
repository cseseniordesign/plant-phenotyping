# How to use Pegasus on HCC server
# SD2019-Plant Phenotyping

## Dependencies
* pegasus (does not need to be installed)
* pandas
* matplotlib
* scipy(version 1.1.0)
* scikit-learn
* pillow
* opencv
* tensorflow
* keras
* numpy

## Cloning Project
1. You can clone the project using: `git clone git@github.com:cseseniordesign/plant-phenotyping.git`

## Transferring Data to HCC Server
2. This workflow can only be run on the HCC server, so move the whole project (plant-phenotyping) into your work directory on the HCC server. Transferring data instructions below:
Mac/Linux (under File Transferring With HCC Supercomputers): <https://hcc.unl.edu/docs/quickstarts/connecting/for_maclinux_users/> 
Window (under File Transferring With HCC Supercomputers): <https://hcc.unl.edu/docs/quickstarts/connecting/for_windows_users/>
3. Ensure that the schnablelab project (<https://github.com/freemao/schnablelab>) is on in the HCC server and is accessible by you.
4. Ensure that the plant dataset is on the HCC server and is accessible by you. 

## Creating Anaconda Environment to Run Workflow
5. Install dependencies: [pandas, matplotlib, scipy(version 1.1.0), scikit-learn, pillow, opencv, tensorflow, keras, and numpy]
    1. Load anaconda: `module load anaconda`
    2. Create your own environment: `conda create -n [env_name]` (Where [env_name] is the name that you want your environment to be) 
    3. To activate the environment: `conda activate [env_name]` (You can verify you have activated your environment by looking to the left username in the terminal. It should show your environment name in parentheses). 
    4. To deactivate environment: `conda deactivate`
    5. Install dependencies [pandas, matplotlib, scipy(version 1.1.0), scikit-learn, pillow, opencv, tensorflow, keras, numpy]: `conda install [package_name]` (must have your environment activated to use this line)

## Modifying Workflow Paths to Work with File Structure
6. Modify rc.txt located in the pics2predictions folder: Each line in rc.txt is just the [file name] [path to file]. Change the path to the model file (.h5) to the path to match to your work directory’s model file. (eg. `file:///work/[groupname]/[username]/[path to project (if applicable)]/plant-phenotyping/pics2predictions/input/model_4_300_3.40421104694 053e-05.h5 site="local-hcc"`
7. Modify tc.txt located in the pics2predictions folder: Change pfn under the tr python3 to match the file path to your work directory’s run_python.sh script. (eg. `file:///work/[groupname]/[username]/[path_to_project (if applicable)]/plant-phenotyping/pics2predictions/scripts/run_python.sh site="local-hcc"`
8. Modify file\_paths\_config.py located in the pics2predictions folder: Change the data value to the file path to the dataset’s Hyp\_SV\_90 folders. Change the model value to the value of the model you want use to predict the images. Change the data\_index value to the value of index of the plant folders name when the data value is separted by /  ex: 
    ` 'data': ‘/work/csesd/pnnguyen/data/*/Hyp_SV_90/’,`
     `'model': ‘model_4_300_3.40421104694053e-05.h5’`
     `'data_index': 5`
    * data needs to be in this structure for workflow to work:
        * data (does not have to be called data)
            * [plant folder name]
                * Hyp\_SV\_90
            * [plant folder name]
                * Hyp\_SV\_90
            * ....
    * Note: There must be just plant folders in the data folder and each plant folder must contain a Hyp\_SV\_90 folder.

9. Modify run_python.sh located in pics2predictions’ scripts folder: Change conda activate numpy to conda activate [env_name] (where [env_name] is the name of the environment you created) and change the python path (content after export PYTHONPATH=) to the path to the schnablelab project. ex: (Note schnablelab folder is in pics2predictions in this example) 
`export PYTHONPATH=/work/csesd/pnnguyen/pics2predictions:$PYTHONPATH`

## Changing Shell Scripts to Executables
10. Run the command: `chmod +x generate_dax.sh` to make generate_dax.sh an executable. 
11. Run the command: `chmod +x plan_dax.sh` to make plan_dax.sh an executable.
12. Run the command: `chmod +x run_python.sh` to make run_python.sh an executable.  

## Running Workflow
13. Run the command: 
`./generate_dax.sh [dax file name].dax` 
(Where [dax file name] is the name you want for the dax file to generate the dax file 
14. Then the command : 
`./plan_dax.sh [dax file name].dax`
to plan the dax and run the workflow
15. Use `pegasus-status -l [copy this from the output]` to see the status. (optional)
ex: `pegasus-status -l /work/csesd/johnsuzh/pics2predictions/submit/johnsuzh/pegasus/split/run0010`
This is to show the status of the workflow.
16. Use `pegasus-remove [copy this from the output]` to remove the current running workflow. (optional)
`pegasus-remove /work/csesd/johnsuzh/pics2predictions/submit/johnsuzh/pegasus/split/run0010`
This is to remove the workflow from the queue.
17. Use `pegasus-analyzer [copy this from the output]` for a failing workflow to see the error message (optional)
`pegasus-analyzer /work/csesd/johnsuzh/pics2predictions/submit/johnsuzh/pegasus/split/run0010`
This is to show the error message of the workflow if there was any error.
18. When it’s done, the status should be 100% in the %DONE column. And you can see the output in the output folder.
19. When you want to run the workflow again, make sure to remove the contents in the output folder
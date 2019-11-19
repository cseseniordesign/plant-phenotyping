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

## Cloning Project (Need to do only once)
1. You can clone the project using: `git clone https://github.com/cseseniordesign/plant-phenotyping.git`
2. Run this command in the location you wish to clone the project to.

## Transferring Data to HCC Server (Need to do only once)
1. This workflow can only be run on the HCC server, so ensure the entire project (plant-phenotyping) is in your work directory on the HCC server. 
2. Ensure that the schnablelab project (<https://github.com/freemao/schnablelab>) is on in the HCC server and is accessible by you.
    1. Run this command, `git clone https://github.com/freemao/schnablelab.git`, in the desired location to add the HCC server.
3. Ensure that the plant dataset is on the HCC server and is accessible by you.
    * Data transfer instructions are below for moving local information to the HCC servers:
        * Mac/Linux (under File Transferring With HCC Supercomputers):<https://hcc.unl.edu/docs/quickstarts/connecting/for_maclinux_users/>
        * Windows (under File Transferring With HCC Supercomputers): <https://hcc.unl.edu/docs/quickstarts/connecting/for_windows_users/>

## Creating Anaconda Environment to Run Workflow (Need to do only once)
1. Install dependencies: [pandas, matplotlib, scipy(version 1.1.0), scikit-learn, pillow, opencv, tensorflow, keras, and numpy]
    1. Load anaconda: `module load anaconda`
    2. Create your own environment: `conda create -n [env_name]` (Where [env_name] is the name that you want your environment to be)
    3. To activate the environment: `conda activate [env_name]` (You can verify you have activated your environment by looking to the left username in the terminal. It should show your environment name in parentheses).
    4. To deactivate environment: `conda deactivate`
    5. Install dependencies [pandas, matplotlib, scikit-learn, pillow, opencv, tensorflow, keras, numpy, scipy(version 1.1.0)]: `conda install [package_name]` (must have your environment activated to use this line)

## Modifying Workflow Paths to Work with File Structure (Need to do only once)
1. Modify file\_paths\_config.py located in the pics2predictions folder: Change the data value to the file path to the dataset’s Hyp\_SV\_90 folders. Change the model value to the value of the model you want use to predict the images (also ensure that the file is in the input folder). Change the data\_index value to the index the plant folder would be at when splitted by /. ex:
    ` 'data': '/work/csesd/pnnguyen/data/*/Hyp_SV_90/',`
     `'model': 'model_4_300_3.40421104694053e-05.h5'`,
     `'data_index': 5`
    * data\_index is 5 here because when data is splitted by /, splitting data above results in ['', work, csesd, pnnguyen, data, * (plant folder name), Hyp\_SV\_90, ''] and because of 0 indexing, * (where the plant folder name is) is the 5th element.
    * data needs to be in this structure for workflow to work:
        * data (does not have to be called data)
            * [plant folder name]
                * Hyp\_SV\_90
            * [plant folder name]
                * Hyp\_SV\_90
            * ....
    * Note: There must be just plant folders in the data folder and each plant folder must contain a Hyp\_SV\_90 folder.

2. Modify run_python.sh located in scripts folder in the pics2predictions folder:
    1. Change conda activate numpy to `conda activate [path to environment]` (where [path to environment] is the path of the environment you created). 
        * Most anaconda paths on the HCC server are either located in $HOME or $WORK. You can check by by typing the command `ls -a` on the command line in either directories and checking if a .conda folder is in the directory. 
       * The actual path to your enviroment is of the following pattern. $HOME/.conda/envs/[name of environment] or $WORK/.conda/envs/[name of environment]. ex:
    `conda activate /home/csesd/pnnguyen/.conda/envs/sd`

    2. Change the python path (content after export PYTHONPATH=) to the path to the schnablelab project. ex: (Note schnablelab folder is in pics2predictions in this example)
`export PYTHONPATH=/work/csesd/pnnguyen/pics2predictions:$PYTHONPATH`

## Changing Shell Scripts to Executables (Need to do only once)
1. Run the command: `chmod +x generate_dax.sh` to make generate_dax.sh an executable.
2. Run the command: `chmod +x plan_dax.sh` to make plan_dax.sh an executable.
3. Run the command: `chmod +x run_python.sh` to make run_python.sh an executable.  

## Running Workflow
1. Run the command:
`./generate_dax.sh [dax file name].dax`
(Where [dax file name] is the name you want for the dax file to generate the dax file
2. Then the command :
`./plan_dax.sh [dax file name].dax`
to plan the dax and run the workflow
3. Use `pegasus-status -l [copy this from the output]` to see the status. (optional)
ex: `pegasus-status -l /work/csesd/johnsuzh/pics2predictions/submit/johnsuzh/pegasus/split/run0010`
This is to show the status of the workflow.
3. Use `pegasus-remove [copy this from the output]` to remove the current running workflow. (optional)
ex:`pegasus-remove /work/csesd/johnsuzh/pics2predictions/submit/johnsuzh/pegasus/split/run0010`
This is to remove the workflow from the queue.
4. Use `pegasus-analyzer [copy this from the output]` for a failing workflow to see the error message (optional)
ex:`pegasus-analyzer /work/csesd/johnsuzh/pics2predictions/submit/johnsuzh/pegasus/split/run0010`
This is to show the error message of the workflow if there was any error.
5. When it’s done, the status should be 100% in the %DONE column (use `pegasus-status` command). And you can see the output in the output folder.
6. When you want to run the workflow again, make sure to remove the contents in the output folder.

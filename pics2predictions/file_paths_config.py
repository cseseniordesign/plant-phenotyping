#!/usr/bin/env python

# data needs to be in this structure workflow to work with hyperspectral images:
# - data (does not have to be called data)
#   - [plant folder name]
#    - Hyp_SV_90
#   - [plant folder name]
#    - Hyp_SV_90
#   ....

# data needs to be in this structure for workflow to work with numpy arrays:
# - data (does not have to be called data)
#   - .npy
#   - .npy
#   ...


# Note: There must be just plant folders in the data folder and each plant folder must contain a Hyp_SV_90 folder.
# Example of data value for hyperspectral images (Plant Folders in test_data_hyp): 'data': '/work/csesd/pnnguyen/plant-phenotyping/pics2predictions/test_data_hyp/*/Hyp_SV_90/'
# Example of data value for numpy arrays (.npy in test_data_npy): 'data': '/work/csesd/pnnguyen/plant-phenotyping/pics2predictions/test_data_npy/*'

file_paths = {'data': '/work/csesd/pnnguyen/plant-phenotyping/pics2predictions/test_data_npy_big/*',
              'model': 'fold3_model_4_300_0.0005.h5'
}

use_anonymous = True

#!/usr/bin/env python

# data needs to be in this structure for daxgen_hyp workflow to work:
# - data (does not have to be called data)
#   - [plant folder name]
#    - Hyp_SV_90
#   - [plant folder name]
#    - Hyp_SV_90
#   ....

# data needs to be in this structure for daxgen_npy workflow to work:
# - data (does not have to be called data)
#   - .npy
#   - .npy
#   ...


# Note: There must be just plant folders in the data folder and each plant folder must contain a Hyp_SV_90 folder.
# Example of data value for daxgen_hyp (Plant Folders in test_data): 'data': '/work/csesd/pnnguyen/pics2predictions/test_data/*/Hyp_SV_90/'
# Example of data value for daxgen_npy: 'data' (.npy in test_data): '/work/csesd/pnnguyen/plant-phenotyping/pics2predictions2/test_data/*'

file_paths = {'data': '/work/csesd/pnnguyen/pics2predictions/test_data/*/Hyp_SV_90/',
              'model': 'model_4_300_3.40421104694053e-05.h5'
}

use_anonymous = True

#!/usr/bin/env python

# data needs to be in this structure for workflow to work:
# - data (does not have to be called data)
#   - [plant folder name]
#    - Hyp_SV_90
#   - [plant folder name]
#    - Hyp_SV_90
#   ....

# Note: There must be just plant folders in the data folder and each plant folder must contain a Hyp_SV_90 folder.

file_paths = {'data': '/work/csesd/pnnguyen/pics2predictions/test_data/*/Hyp_SV_90/',
              'model': 'model_4_300_3.40421104694053e-05.h5'
}

use_anonymous = True

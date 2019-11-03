#!/bin/bash

. /util/opt/lmod/lmod/init/profile
export -f module
module use /util/opt/hcc-modules/Common/

module load anaconda
conda activate mynumpy

export PYTHONPATH=/work/csesd/johnsuzh/pic2prediction:$PYTHONPATH
python3 "$@"

conda deactivate

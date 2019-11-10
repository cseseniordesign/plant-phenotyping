#!/bin/bash

. /util/opt/lmod/lmod/init/profile
export -f module
module use /util/opt/hcc-modules/Common/

module load anaconda
conda activate mynumpy

export PYTHONPATH=/work/csesd/johnsuzh/pics2predictions:$PYTHONPATH
python3 "$@"

conda deactivate

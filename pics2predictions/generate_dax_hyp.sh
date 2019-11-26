#!/bin/bash
set -e
export PYTHONPATH=$(pegasus-config --python)
exec python daxgen_hyp.py "$@"

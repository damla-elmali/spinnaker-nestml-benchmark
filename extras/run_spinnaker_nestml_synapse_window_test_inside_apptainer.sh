#!/bin/bash

cd /users/elmali/nestml

# Create this directory first so it is included in PYTHONPATH
mkdir -p /users/elmali/nestml/spinnaker-install

export PATH=/users/elmali/.local/bin:$PATH
export PYTHONPATH=/users/elmali/nestml/spinnaker-install:$PYTHONPATH

python3 -m pytest \
    -o log_cli=true \
    -o log_cli_level="DEBUG" \
    -s \
    --pdb \
    /users/elmali/nestml/tests/spinnaker_tests/test_spinnaker_stdp_window.py
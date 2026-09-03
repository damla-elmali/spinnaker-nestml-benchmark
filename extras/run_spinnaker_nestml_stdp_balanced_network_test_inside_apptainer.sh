#!/bin/bash

# Run the NESTML balanced-network test on the SpiNNaker backend.
# This script is intended to be executed inside the Apptainer environment.
# It prepares the generated model installation path and runs the test with pytest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NESTML_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mv -v "$NESTML_ROOT/spinnaker-target" "$NESTML_ROOT/spinnaker-target-$(date +%Y-%m-%d_%H-%M-%S.%N | cut -b1-23)"
mv -v "$NESTML_ROOT/spinnaker-install" "$NESTML_ROOT/spinnaker-install-$(date +%Y-%m-%d_%H-%M-%S.%N | cut -b1-23)"

cd "$NESTML_ROOT" || exit 1

# Create the installation directory for the generated SpiNNaker code.
mkdir -p "$NESTML_ROOT/spinnaker-install"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$NESTML_ROOT/spinnaker-install:$PYTHONPATH"

export SPINNAKER_DIRS=/home/spinnaker/source/spinnaker_tools
export NEURAL_MODELLING_DIRS=/home/spinnaker/source/sPyNNaker/neural_modelling
export SPINN_COMMON_INSTALL_DIR=/home/spinnaker/source/spinn_common


pytest -s -o log_cli=true -o log_cli_level="DEBUG" "$NESTML_ROOT/tests/spinnaker_tests/test_spinnaker_damla_balanced_network_stdp.py"
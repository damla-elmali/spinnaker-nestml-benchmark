#!/bin/bash

# Run the NESTML balanced-network test on the SpiNNaker backend.
# This script is intended to be executed inside the Apptainer environment.
# It prepares the generated model installation path and runs the test with pytest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NESTML_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$NESTML_ROOT" || exit 1

# Create the installation directory for the generated SpiNNaker code.
mkdir -p "$NESTML_ROOT/spinnaker-install"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$NESTML_ROOT/spinnaker-install:$PYTHONPATH"

pytest -s -o log_cli=true -o log_cli_level="DEBUG" "$NESTML_ROOT/tests/spinnaker_tests/test_spinnaker_damla_balanced_network.py"
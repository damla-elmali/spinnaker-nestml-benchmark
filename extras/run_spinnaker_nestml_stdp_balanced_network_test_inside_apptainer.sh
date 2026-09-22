#!/bin/bash

# Run the NESTML balanced-network test on the SpiNNaker backend.
# This script is intended to be executed inside the Apptainer environment.
# It prepares the generated model installation path and runs the test with pytest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NESTML_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_FILE="$NESTML_ROOT/tests/spinnaker_tests/test_spinnaker_balanced_network.py"

mv -v "$NESTML_ROOT/spinnaker-target" "$NESTML_ROOT/spinnaker-target-$(date +%Y-%m-%d_%H-%M-%S.%N | cut -b1-23)"
mv -v "$NESTML_ROOT/spinnaker-install" "$NESTML_ROOT/spinnaker-install-$(date +%Y-%m-%d_%H-%M-%S.%N | cut -b1-23)"

cd "$NESTML_ROOT" || exit 1

# Create the installation directory for the generated SpiNNaker code.
mkdir -p "$NESTML_ROOT/spinnaker-install"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$NESTML_ROOT/spinnaker-install:$NESTML_ROOT/extras/spinnaker1:$PYTHONPATH"

export SPINNAKER_DIRS=/home/spinnaker/source/spinnaker_tools
export NEURAL_MODELLING_DIRS=/home/spinnaker/source/sPyNNaker/neural_modelling
export SPINN_COMMON_INSTALL_DIR=/home/spinnaker/source/spinn_common


case "$1" in
    comparison)
        export USE_EXP_LUTS=true
        pytest -s -o log_cli=true -o log_cli_level="DEBUG" \
            "$TEST_FILE::TestSpiNNakerBalancedNetwork::test_spinnaker_balanced_network"
        ;;

    nestml)
        export USE_EXP_LUTS=false
        pytest -s -o log_cli=true -o log_cli_level="DEBUG" \
            "$TEST_FILE::TestSpiNNakerBalancedNetwork::test_nestml"
        ;;

    *)
        echo "Usage: $0 {comparison|nestml}"
        exit 1
        ;;
esac
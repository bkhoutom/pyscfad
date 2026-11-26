#!/usr/bin/env bash

set -euo pipefail
sudo apt-get -qq install libopenblas-dev
export CMAKE_BUILD_PARALLEL_LEVEL=2
python -m pip install ./pyscfadlib
python -m pip install -e .

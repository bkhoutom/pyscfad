#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/run_water_16_ccsd_t.sh" \
  --waters 10 --xyz "$script_dir/water_10.xyz" \
  --output-dir output/water_10_ccpvtz_stc "$@"

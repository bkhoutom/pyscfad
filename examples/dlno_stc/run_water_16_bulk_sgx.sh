#!/usr/bin/env bash
# Bulk-carved water16 from the completed 2026-11-10 SGX force benchmark.
# Deterministic W, CCSD and (T), with three fitted (T) Laplace points per LIS.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"

export OMP_DYNAMIC=TRUE

# The shared runner configures CPU affinity and numerical threads before imports.
exec "$python_bin" -u "$script_dir/benchmark_lno_w_triples.py" \
  --xyz-file "$script_dir/water_16_bulk.xyz" \
  --basis cc-pvtz --frozen 16 --boys-grouping atom \
  --domain-thresholds "$script_dir/water_16_bulk_thresholds.json" \
  --lno-thresh-occ 1e-4 --lno-thresh-vir 3e-5 \
  --scf-jk dfj-sgx --sgx-integral-backend generated \
  --sgx-grid-level 2 --sgx-fit-overlap --sgx-screening per_pair \
  --sgx-mixed-precision --sgx-energy-tol 1e-10 --sgx-potential-tol 1e-8 \
  --sgx-fp32-multiplier 1e5 \
  --threads "${THREADS:-32}" \
  --max-memory-mb "${MAX_MEMORY_MB:-900000}" \
  --df-memory-mb "${DF_MEMORY_MB:-500000}" \
  --fragment-df-mode half --triples-integral-transform rotate \
  --deterministic-only --triples-laplace-points 3 \
  --cc-conv-tol 1e-7 --cc-conv-tol-normt 1e-5 \
  --output-dir output/water_16_bulk_sgx "$@"

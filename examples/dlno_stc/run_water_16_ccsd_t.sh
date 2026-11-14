#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
threads="${THREADS:-4}"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
mpi_exec="${MPIEXEC:-mpiexec}"

# Configure all numerical runtimes before Python/JAX/BLAS are imported.
export OMP_NUM_THREADS="$threads" OMP_THREAD_LIMIT="$threads"
export OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYSCFAD_DENSE_BLAS_THREADS="$threads"
# JAX's CPU runtime uses its own worker pools, independent of OpenMP/BLAS.
export PJRT_NPROC="$threads"
# Low-memory eager execution also makes the stage wall-time boundaries precise.
export JAX_CPU_ENABLE_ASYNC_DISPATCH="${JAX_CPU_ENABLE_ASYNC_DISPATCH:-false}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false}"
export PYSCFAD_STC_CLEAR_JAX_CACHES="${PYSCFAD_STC_CLEAR_JAX_CACHES:-1}"
# Keep the measured low-memory input policy; opt into resident H reuse with half.
export PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE="${PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE:-direct}"
export PYSCFAD_SCF_OCCUPIED_CACHE="${PYSCFAD_SCF_OCCUPIED_CACHE:-0}"
# Open MPI otherwise may bind the sole rank to one core. Other MPI implementations ignore this.
export OMPI_MCA_hwloc_base_binding_policy=none

# Ignore inherited PYTHONPATH/PYTHONHOME and user-site packages. In particular,
# an older neighboring PySCF checkout must not shadow this environment's PySCF.
exec "$mpi_exec" -n 1 "$python_bin" -I -u "$script_dir/water_16_ccsd_t_gradient.py" --threads "$threads" "$@"

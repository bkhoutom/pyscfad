#!/usr/bin/env bash
#SBATCH --job-name=xing-pyscfad
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=reproduce_dcmp_ad.out.16.3.%j
#SBATCH --error=reproduce_dcmp_ad.err.16.3.%j

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from dCMP}"
PYTHON=${PYTHON_BIN:-/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python}
export PATH="${PYTHON%/*}:$PATH"
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1

SCRATCH_DIR="/resnick/scratch/$USER/xing/${SLURM_JOB_ID:?Run via sbatch}"
mkdir -p "$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR" PYSCF_TMPDIR="$SCRATCH_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_THREAD_LIMIT="$OMP_NUM_THREADS" OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE MKL_DYNAMIC=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
# Keep BLAS single-threaded inside PySCF OpenMP regions.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# Only DF construction and its pullback use the larger SciPy BLAS pool.
export SCIPY_DF_THREADS="$SLURM_CPUS_PER_TASK"
# Balance integral derivative work across OpenMP threads.
export PYSCFAD_INTEGRAL_VJP_SCHEDULE=dynamic
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True JAX_ENABLE_COMPILATION_CACHE=false

srun --exact --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --distribution=block:block --cpu-bind=cores "$PYTHON" -u reproduce_dcmp_ad.py

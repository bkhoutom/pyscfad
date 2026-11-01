#!/usr/bin/env bash
#SBATCH --job-name=water-4-LT
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this water directory}"
PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
export PATH="${PYTHON%/*}:$PATH" PYTHONNOUSERSITE=1
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE

SCRATCH_DIR="/resnick/scratch/$USER/water_LT/${SLURM_JOB_ID:?Run via sbatch}"
mkdir -p "$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR" PYSCF_TMPDIR="$SCRATCH_DIR"

# Native OpenMP uses the allocation; nested BLAS remains single-threaded.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_THREAD_LIMIT="$OMP_NUM_THREADS" OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE MKL_DYNAMIC=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
# The local driver raises only SciPy's pool during DF forward and reverse.
export SCIPY_DF_THREADS="$SLURM_CPUS_PER_TASK"
export PYSCFAD_INTEGRAL_VJP_SCHEDULE=dynamic
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True JAX_ENABLE_COMPILATION_CACHE=false
export PYSCFAD_DLNO_RESOURCE_PROFILE=1

srun --exact --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --distribution=block:block --cpu-bind=cores "$PYTHON" -u compare_mp2_LT.py

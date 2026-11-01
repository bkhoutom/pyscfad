#!/usr/bin/env bash
#SBATCH --job-name=dcmp-scf-df-split
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this condition directory}"
PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
export PATH="${PYTHON%/*}:$PATH" PYTHONNOUSERSITE=1
unset LD_LIBRARY_PATH OPENBLAS_CORETYPE
export OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE MKL_DYNAMIC=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export SCIPY_DF_THREADS=16 PYSCFAD_DENSE_BLAS_THREADS=16
export PYSCFAD_FACTOR_PANEL_MEMORY_MB=4096 PYSCFAD_SOLVE_MEMORY_MB=64
export PYSCFAD_PRECOND_CACHE_MEMORY_MB=1024
export PYSCFAD_INTEGRAL_VJP_SCHEDULE=dynamic PYSCFAD_PROFILE_BACKWARD_PHASES=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True JAX_ENABLE_COMPILATION_CACHE=false
export PYTHONPATH="$PWD/code" BENCH_CASE=optimized_ooc_split BENCH_MODE=outcore
export TMPDIR="/resnick/scratch/$USER/dcmp-reverse/$SLURM_JOB_ID/$BENCH_CASE"
export PYSCF_TMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"
hostname
srun --exact --ntasks=1 --cpus-per-task=16 --distribution=block:block --cpu-bind=cores "$PYTHON" -u driver.py

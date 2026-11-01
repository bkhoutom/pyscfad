#!/usr/bin/env bash
#SBATCH --job-name=reverse-red-precond
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=debug
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this condition directory}"
PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
export PATH="${PYTHON%/*}:$PATH" PYTHONNOUSERSITE=1
unset LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONPATH="$PWD/code"
export TMPDIR="/resnick/scratch/$USER/dcmp-reverse/$SLURM_JOB_ID"
export PYSCF_TMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4 OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE MKL_DYNAMIC=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYSCFAD_DENSE_BLAS_THREADS=4
export PYSCFAD_INTEGRAL_VJP_SCHEDULE=dynamic
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True JAX_ENABLE_COMPILATION_CACHE=false
srun --exact --ntasks=1 --cpus-per-task=4 --distribution=block:block --cpu-bind=cores "$PYTHON" -u driver.py

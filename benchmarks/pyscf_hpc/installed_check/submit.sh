#!/usr/bin/env bash
#SBATCH --job-name=pyscf-installed-check
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=debug
#SBATCH --constraint=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this directory}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_THREAD_LIMIT="$SLURM_CPUS_PER_TASK"
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset PYTHONPATH
mkdir -p scratch
export TMPDIR="$PWD/scratch"
export XDG_CACHE_HOME="$PWD/scratch/cache"

hostname
lscpu
srun --cpu-bind=cores /resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python -u driver.py

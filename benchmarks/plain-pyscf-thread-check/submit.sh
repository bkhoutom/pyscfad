#!/usr/bin/env bash
#SBATCH --job-name=plain-dcmp-df-threads
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=debug
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this directory}"
PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE MKL_DYNAMIC=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export SCIPY_DF_THREADS=16
mkdir -p scratch
export TMPDIR="$PWD/scratch" PYSCF_TMPDIR="$PWD/scratch"
srun --exact --ntasks=1 --cpus-per-task=16 --distribution=block:block --cpu-bind=cores "$PYTHON" -u driver.py original > original.out 2> original.err
srun --exact --ntasks=1 --cpus-per-task=16 --distribution=block:block --cpu-bind=cores "$PYTHON" -u driver.py optimized > optimized.out 2> optimized.err

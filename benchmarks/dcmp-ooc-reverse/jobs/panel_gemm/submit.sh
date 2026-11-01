#!/usr/bin/env bash
#SBATCH --job-name=panel-gemm
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:10:00
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
srun --exact --ntasks=1 --cpus-per-task=16 --distribution=block:block --cpu-bind=cores "$PYTHON" -u driver.py

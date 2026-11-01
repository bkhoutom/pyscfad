#!/usr/bin/env bash
#SBATCH --job-name=final-native-check
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this condition directory}"
PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_THREAD_LIMIT="$SLURM_CPUS_PER_TASK"
export OMP_MAX_ACTIVE_LEVELS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cpu JAX_ENABLE_COMPILATION_CACHE=false
mkdir -p scratch
export TMPDIR="$PWD/scratch" PYSCF_TMPDIR="$PWD/scratch"
srun --cpu-bind=cores bash build.sh > build.log 2>&1
srun --cpu-bind=cores "$PYTHON" -u driver.py

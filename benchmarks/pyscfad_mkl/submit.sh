#!/usr/bin/env bash
#SBATCH --job-name=fork-mkl-package
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this condition directory}"
PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_THREAD_LIMIT="$SLURM_CPUS_PER_TASK"
export OMP_MAX_ACTIVE_LEVELS=1 OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cpu JAX_ENABLE_COMPILATION_CACHE=false
mkdir -p scratch results wheels rollback
export TMPDIR="$PWD/scratch" PYSCF_TMPDIR="$PWD/scratch" XDG_CACHE_HOME="$PWD/scratch/cache"

srun --cpu-bind=cores bash build.sh > build.log 2>&1
srun --cpu-bind=cores "$PYTHON" -u driver.py
"$PYTHON" package_build.py
